"""Train a separate recent low-FPR Poker44 model as a hedge strategy."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
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


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    params: dict[str, Any]
    human_weight: float
    recency_halflife_days: float
    factory: Callable[[], Any]


@dataclass
class FittedCandidate:
    spec: CandidateSpec
    model: Any
    validation_scores: np.ndarray
    validation_metrics: dict[str, Any]


def available_feature_dates(data_dir: Path) -> list[str]:
    return sorted(
        child.name
        for child in data_dir.iterdir()
        if child.is_dir() and (child / "features.csv").exists()
    )


def feature_paths(data_dir: Path, start: str, end: str) -> list[Path]:
    dates = [date for date in available_feature_dates(data_dir) if start <= date <= end]
    if not dates:
        raise RuntimeError(f"No feature CSVs found between {start} and {end}")
    return [data_dir / date / "features.csv" for date in dates]


def load_feature_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise RuntimeError(f"{path} does not contain a label column")
    return frame


def load_feature_csvs(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([load_feature_csv(path) for path in paths], ignore_index=True, sort=False)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not columns:
        raise RuntimeError("No numeric feature columns found")
    return columns


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} CSV is missing feature columns: {missing}")


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def evaluate_scores(scores: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    subnet_reward, metrics = reward(scores, y_true)
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": _safe_roc_auc(y_true, scores),
        "subnet_reward": float(subnet_reward),
        **{key: float(value) for key, value in metrics.items()},
    }


def model_scores(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    scores = np.asarray(model.decision_function(x), dtype=float)
    return 1.0 / (1.0 + np.exp(-scores))


def sample_weights(frame: pd.DataFrame, y_true: np.ndarray, spec: CandidateSpec) -> np.ndarray:
    weights = np.where(y_true == 0, float(spec.human_weight), 1.0).astype(float)
    if spec.recency_halflife_days > 0:
        dates = pd.to_datetime(frame["source_date"], errors="coerce")
        max_date = dates.max()
        ages = (max_date - dates).dt.days.fillna(0).clip(lower=0).astype(float).to_numpy()
        recency = 0.35 + 0.65 * np.exp(-ages / float(spec.recency_halflife_days))
        weights *= recency
    return weights / max(float(np.mean(weights)), 1e-12)


def make_hgb_spec(
    *,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    l2_regularization: float,
    min_samples_leaf: int,
    human_weight: float,
    recency_halflife_days: float,
    random_state: int,
) -> CandidateSpec:
    params = {
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "l2_regularization": l2_regularization,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
    }
    name = (
        f"human_safe_hgb_lr{learning_rate}_iter{max_iter}_leaf{max_leaf_nodes}"
        f"_l2{l2_regularization}_min{min_samples_leaf}_hw{human_weight}_half{recency_halflife_days}"
    )
    return CandidateSpec(
        name=name,
        family="hist_gradient_boosting",
        params=params,
        human_weight=human_weight,
        recency_halflife_days=recency_halflife_days,
        factory=lambda params=params: HistGradientBoostingClassifier(**params),
    )


def make_extra_trees_spec(
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str | float,
    human_weight: float,
    recency_halflife_days: float,
    random_state: int,
) -> CandidateSpec:
    params = {
        "n_estimators": n_estimators,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
    }
    name = (
        f"human_safe_extra_trees_n{n_estimators}_min{min_samples_leaf}_feat{max_features}"
        f"_hw{human_weight}_half{recency_halflife_days}"
    )
    return CandidateSpec(
        name=name,
        family="extra_trees",
        params=params,
        human_weight=human_weight,
        recency_halflife_days=recency_halflife_days,
        factory=lambda params=params: ExtraTreesClassifier(**params),
    )


def build_specs(random_state: int, families: set[str]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    human_weights = (1.0, 1.7, 2.6)
    recency_halflives = (0.0, 7.0)

    if "hgb" in families:
        hgb_shapes = (
            (0.03, 180, 7, 0.05, 10),
            (0.05, 160, 7, 0.05, 10),
            (0.03, 220, 15, 0.05, 15),
        )
        for human_weight in human_weights:
            for halflife in recency_halflives:
                for shape in hgb_shapes:
                    specs.append(
                        make_hgb_spec(
                            learning_rate=shape[0],
                            max_iter=shape[1],
                            max_leaf_nodes=shape[2],
                            l2_regularization=shape[3],
                            min_samples_leaf=shape[4],
                            human_weight=human_weight,
                            recency_halflife_days=halflife,
                            random_state=random_state,
                        )
                    )

    if "extra_trees" in families:
        extra_trees_shapes = (
            (300, 3, 0.5),
            (300, 5, "sqrt"),
        )
        for human_weight in human_weights:
            for halflife in (0.0, 7.0):
                for shape in extra_trees_shapes:
                    specs.append(
                        make_extra_trees_spec(
                            n_estimators=shape[0],
                            min_samples_leaf=shape[1],
                            max_features=shape[2],
                            human_weight=human_weight,
                            recency_halflife_days=halflife,
                            random_state=random_state,
                        )
                    )

    return specs


def fit_candidate(
    spec: CandidateSpec,
    train_frame: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> FittedCandidate:
    model = spec.factory()
    model.fit(x_train, y_train, sample_weight=sample_weights(train_frame, y_train, spec))
    scores = model_scores(model, x_validation)
    return FittedCandidate(
        spec=spec,
        model=model,
        validation_scores=scores,
        validation_metrics=evaluate_scores(scores, y_validation),
    )


def sort_key(candidate: FittedCandidate) -> tuple[float, float, float, float]:
    metrics = candidate.validation_metrics
    return (
        float(metrics["subnet_reward"]),
        float(metrics["bot_recall"]),
        -float(metrics["fpr"]),
        float(metrics["average_precision"]),
    )


def serialize_spec(spec: CandidateSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "family": spec.family,
        "params": spec.params,
        "human_weight": spec.human_weight,
        "recency_halflife_days": spec.recency_halflife_days,
    }


def save_bundle(
    *,
    output_path: Path,
    model_name: str,
    model: Any,
    columns: list[str],
    train_paths: list[Path],
    selected: FittedCandidate,
    validation_path: Path | None,
    validation_metrics: dict[str, Any] | None,
    notes: str,
) -> None:
    bundle = {
        "model_name": model_name,
        "model": model,
        "feature_columns": columns,
        "train_paths": [str(path) for path in train_paths],
        "validation_path": str(validation_path) if validation_path is not None else None,
        "selection_metric": "validation.subnet_reward",
        "selected_strategy": serialize_spec(selected.spec),
        "validation": validation_metrics,
        "notes": notes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def parse_families(values: list[str]) -> set[str]:
    aliases = {"hgb": "hgb", "hist_gradient_boosting": "hgb", "extra_trees": "extra_trees"}
    families = {aliases.get(value, value) for value in values}
    unknown = sorted(families - {"hgb", "extra_trees"})
    if unknown:
        raise RuntimeError(f"Unknown families: {unknown}")
    return families


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="downloads/poker44_benchmark")
    parser.add_argument("--train-start", default="2026-07-06")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--final-output", default="")
    parser.add_argument("--random-state", type=int, default=144)
    parser.add_argument("--families", nargs="+", default=["hgb", "extra_trees"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_paths = feature_paths(data_dir, args.train_start, args.train_end)
    validation_path = data_dir / args.validation_date / "features.csv"
    train_frame = load_feature_csvs(train_paths)
    validation_frame = load_feature_csv(validation_path)

    columns = feature_columns(train_frame)
    require_columns(validation_frame, columns, "Validation")
    x_train = train_frame[columns].fillna(0.0)
    y_train = train_frame["label"].astype(int).to_numpy()
    x_validation = validation_frame[columns].fillna(0.0)
    y_validation = validation_frame["label"].astype(int).to_numpy()

    specs = build_specs(args.random_state, parse_families(args.families))
    if not specs:
        raise RuntimeError("No candidate specs were built")

    candidates: list[FittedCandidate] = []
    for index, spec in enumerate(specs, start=1):
        candidates.append(
            fit_candidate(spec, train_frame, x_train, y_train, x_validation, y_validation)
        )
        if index % 20 == 0 or index == len(specs):
            print(f"fit {index}/{len(specs)} low-FPR candidates", flush=True)

    ranked = sorted(candidates, key=sort_key, reverse=True)
    selected = ranked[0]
    output_path = Path(args.output)
    save_bundle(
        output_path=output_path,
        model_name=f"{selected.spec.name}_train_to_{args.train_end}",
        model=selected.model,
        columns=columns,
        train_paths=train_paths,
        selected=selected,
        validation_path=validation_path,
        validation_metrics=selected.validation_metrics,
        notes=(
            "Recent low-FPR specialist selected as a second strategy. "
            "This is intentionally separate from the main ensemble model."
        ),
    )

    final_output = None
    if args.final_output:
        final_paths = feature_paths(data_dir, args.train_start, args.validation_date)
        final_frame = load_feature_csvs(final_paths)
        require_columns(final_frame, columns, "Final training")
        x_final = final_frame[columns].fillna(0.0)
        y_final = final_frame["label"].astype(int).to_numpy()
        final_model = selected.spec.factory()
        final_model.fit(
            x_final,
            y_final,
            sample_weight=sample_weights(final_frame, y_final, selected.spec),
        )
        final_output_path = Path(args.final_output)
        save_bundle(
            output_path=final_output_path,
            model_name=f"{selected.spec.name}_final_through_{args.validation_date}",
            model=final_model,
            columns=columns,
            train_paths=final_paths,
            selected=selected,
            validation_path=None,
            validation_metrics=None,
            notes=(
                "Final refit of the selected recent low-FPR specialist through the "
                f"{args.validation_date} public benchmark release."
            ),
        )
        final_output = str(final_output_path)

    print(
        json.dumps(
            {
                "selected_output": str(output_path),
                "final_output": final_output,
                "feature_count": len(columns),
                "train_start": args.train_start,
                "train_end": args.train_end,
                "validation_date": args.validation_date,
                "train_rows": int(len(train_frame)),
                "validation_rows": int(len(validation_frame)),
                "label_counts": {
                    "train_human": int(np.sum(y_train == 0)),
                    "train_bot": int(np.sum(y_train == 1)),
                    "validation_human": int(np.sum(y_validation == 0)),
                    "validation_bot": int(np.sum(y_validation == 1)),
                },
                "selected": {
                    **serialize_spec(selected.spec),
                    "validation": selected.validation_metrics,
                },
                "top_candidates": [
                    {
                        **serialize_spec(candidate.spec),
                        "validation": candidate.validation_metrics,
                    }
                    for candidate in ranked[:10]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
