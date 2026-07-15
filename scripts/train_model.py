"""Train baseline Poker44 bot-detection models from feature CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from poker44.score.scoring import reward


METADATA_COLUMNS = {
    "source_date",
    "split",
    "chunk_id",
    "chunk_hash",
    "chunk_index",
    "group_index",
    "label",
    "label_name",
}


def load_feature_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise RuntimeError(f"{path} does not contain a label column")
    return frame


def load_feature_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [load_feature_csv(path) for path in paths]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True, sort=False)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not columns:
        raise RuntimeError("No numeric feature columns found")
    return columns


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def evaluate_model(model: Any, x: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1]
    else:
        scores = model.decision_function(x)

    scores = np.asarray(scores, dtype=float)
    subnet_reward, metrics = reward(scores, y)
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": _safe_roc_auc(y, scores),
        "subnet_reward": float(subnet_reward),
        **{key: float(value) for key, value in metrics.items()},
    }


def build_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier  # type: ignore

        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=15,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbose=-1,
        )
    except Exception:
        pass

    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        required=True,
        nargs="+",
        help="One or more training feature CSVs",
    )
    parser.add_argument("--test", required=True, help="Test feature CSV")
    parser.add_argument("--output", default="models/poker44_baseline.joblib")
    parser.add_argument("--random-state", type=int, default=44)
    args = parser.parse_args()

    train_paths = [Path(path) for path in args.train]
    train_frame = load_feature_csvs(train_paths)
    test_frame = load_feature_csv(Path(args.test))
    columns = feature_columns(train_frame)

    missing = sorted(set(columns) - set(test_frame.columns))
    if missing:
        raise RuntimeError(f"Test CSV is missing feature columns: {missing}")

    x_train = train_frame[columns].fillna(0.0)
    y_train = train_frame["label"].astype(int).to_numpy()
    x_test = test_frame[columns].fillna(0.0)
    y_test = test_frame["label"].astype(int).to_numpy()

    results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}
    for name, model in build_models(args.random_state).items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        results[name] = {
            "train": evaluate_model(model, x_train, y_train),
            "test": evaluate_model(model, x_test, y_test),
        }

    best_name = max(
        results,
        key=lambda name: (
            float(results[name]["test"]["subnet_reward"]),
            float(results[name]["test"]["average_precision"]),
        ),
    )
    bundle = {
        "model_name": best_name,
        "model": fitted_models[best_name],
        "feature_columns": columns,
        "train_path": [str(path) for path in train_paths],
        "test_path": args.test,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)

    print(
        json.dumps(
            {
                "best_model": best_name,
                "output": str(output_path),
                "feature_count": len(columns),
                "train_paths": [str(path) for path in train_paths],
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
