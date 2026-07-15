"""Tune Poker44 bot-detection models with a held-out validation day."""

from __future__ import annotations

import argparse
import copy
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from poker44.modeling import ProbabilityAveragingEnsemble
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
    model_type: str
    params: dict[str, Any]
    weight_mode: str
    factory: Callable[[], Any]


@dataclass
class FittedCandidate:
    spec: CandidateSpec
    model: Any
    validation_scores: np.ndarray
    test_scores: np.ndarray
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


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


def evaluate_scores(scores: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    subnet_reward, metrics = reward(scores, y)
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": _safe_roc_auc(y, scores),
        "subnet_reward": float(subnet_reward),
        **{key: float(value) for key, value in metrics.items()},
    }


def model_scores(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    scores = np.asarray(model.decision_function(x), dtype=float)
    return 1.0 / (1.0 + np.exp(-scores))


def recent_weights(frame: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None

    dates = pd.to_datetime(frame["source_date"], errors="coerce")
    max_date = dates.max()
    ages = (max_date - dates).dt.days.fillna(0).clip(lower=0).astype(float).to_numpy()

    if mode == "recent_soft":
        weights = 0.45 + 0.55 * np.exp(-ages / 10.0)
    elif mode == "recent_strong":
        weights = 0.20 + 0.80 * np.exp(-ages / 5.0)
    elif mode == "full_release_boost":
        weights = np.where(frame["source_date"].astype(str) >= "2026-07-06", 1.0, 0.45)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    return weights / weights.mean()


def fit_spec(spec: CandidateSpec, x: pd.DataFrame, y: np.ndarray, frame: pd.DataFrame) -> Any:
    model = spec.factory()
    weights = recent_weights(frame, spec.weight_mode)
    if weights is None:
        model.fit(x, y)
    else:
        model.fit(x, y, sample_weight=weights)
    return model


def make_hgb_spec(
    *,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    l2_regularization: float,
    min_samples_leaf: int,
    weight_mode: str,
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
        f"hgb_lr{learning_rate}_iter{max_iter}_leaf{max_leaf_nodes}"
        f"_l2{l2_regularization}_min{min_samples_leaf}_{weight_mode}"
    )
    return CandidateSpec(
        name=name,
        model_type="hist_gradient_boosting",
        params=params,
        weight_mode=weight_mode,
        factory=lambda params=params: HistGradientBoostingClassifier(**params),
    )


def make_rf_spec(
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str | float,
    max_depth: int | None,
    weight_mode: str,
    random_state: int,
) -> CandidateSpec:
    params = {
        "n_estimators": n_estimators,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "max_depth": max_depth,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
    }
    name = (
        f"rf_n{n_estimators}_min{min_samples_leaf}_feat{max_features}"
        f"_depth{max_depth}_{weight_mode}"
    )
    return CandidateSpec(
        name=name,
        model_type="random_forest",
        params=params,
        weight_mode=weight_mode,
        factory=lambda params=params: RandomForestClassifier(**params),
    )


def make_extra_trees_spec(
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str | float,
    weight_mode: str,
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
    name = f"extra_trees_n{n_estimators}_min{min_samples_leaf}_feat{max_features}_{weight_mode}"
    return CandidateSpec(
        name=name,
        model_type="extra_trees",
        params=params,
        weight_mode=weight_mode,
        factory=lambda params=params: ExtraTreesClassifier(**params),
    )


def make_xgboost_spec(
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    min_child_weight: float,
    weight_mode: str,
    random_state: int,
) -> CandidateSpec:
    from xgboost import XGBClassifier  # type: ignore

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_lambda": reg_lambda,
        "min_child_weight": min_child_weight,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": 1,
    }
    name = (
        f"xgb_n{n_estimators}_d{max_depth}_lr{learning_rate}"
        f"_sub{subsample}_col{colsample_bytree}_l{reg_lambda}_child{min_child_weight}_{weight_mode}"
    )
    return CandidateSpec(
        name=name,
        model_type="xgboost",
        params=params,
        weight_mode=weight_mode,
        factory=lambda params=params: XGBClassifier(**params),
    )


def make_lightgbm_spec(
    *,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    min_child_samples: int,
    reg_lambda: float,
    feature_fraction: float,
    bagging_fraction: float,
    weight_mode: str,
    random_state: int,
) -> CandidateSpec:
    from lightgbm import LGBMClassifier  # type: ignore

    params = {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "reg_lambda": reg_lambda,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": 1,
        "random_state": random_state,
        "verbose": -1,
        "n_jobs": 1,
    }
    name = (
        f"lgbm_n{n_estimators}_leaves{num_leaves}_lr{learning_rate}"
        f"_min{min_child_samples}_l{reg_lambda}_feat{feature_fraction}_bag{bagging_fraction}_{weight_mode}"
    )
    return CandidateSpec(
        name=name,
        model_type="lightgbm",
        params=params,
        weight_mode=weight_mode,
        factory=lambda params=params: LGBMClassifier(**params),
    )


def build_specs(random_state: int) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    weight_modes = ("none", "recent_soft", "full_release_boost")

    for weight_mode in weight_modes:
        for learning_rate in (0.03, 0.05, 0.08):
            for max_iter in (200, 350):
                for max_leaf_nodes in (7, 15):
                    for l2_regularization in (0.0, 0.05):
                        specs.append(
                            make_hgb_spec(
                                learning_rate=learning_rate,
                                max_iter=max_iter,
                                max_leaf_nodes=max_leaf_nodes,
                                l2_regularization=l2_regularization,
                                min_samples_leaf=10,
                                weight_mode=weight_mode,
                                random_state=random_state,
                            )
                        )

    for weight_mode in ("none", "recent_soft"):
        for min_samples_leaf in (1, 2, 3):
            for max_features in ("sqrt", 0.5):
                specs.append(
                    make_rf_spec(
                        n_estimators=700,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        max_depth=None,
                        weight_mode=weight_mode,
                        random_state=random_state,
                    )
                )

        for min_samples_leaf in (1, 2):
            for max_features in ("sqrt", 0.5):
                specs.append(
                    make_extra_trees_spec(
                        n_estimators=700,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        weight_mode=weight_mode,
                        random_state=random_state,
                    )
                )

    try:
        import xgboost  # noqa: F401

        for weight_mode in ("none", "recent_soft", "full_release_boost"):
            for params in (
                (200, 2, 0.04, 0.90, 0.75, 2.0, 1.0),
                (300, 2, 0.03, 0.90, 0.90, 5.0, 1.0),
                (350, 3, 0.03, 0.85, 0.75, 3.0, 2.0),
                (450, 3, 0.02, 0.90, 0.90, 5.0, 1.0),
                (250, 4, 0.04, 0.80, 0.75, 8.0, 3.0),
            ):
                specs.append(
                    make_xgboost_spec(
                        n_estimators=params[0],
                        max_depth=params[1],
                        learning_rate=params[2],
                        subsample=params[3],
                        colsample_bytree=params[4],
                        reg_lambda=params[5],
                        min_child_weight=params[6],
                        weight_mode=weight_mode,
                        random_state=random_state,
                    )
                )
    except Exception:
        pass

    try:
        import lightgbm  # noqa: F401

        for weight_mode in ("none", "recent_soft", "full_release_boost"):
            for params in (
                (250, 7, 0.04, 8, 1.0, 0.75, 0.90),
                (350, 7, 0.03, 12, 3.0, 0.90, 0.90),
                (300, 15, 0.03, 10, 2.0, 0.75, 0.85),
                (450, 15, 0.02, 16, 5.0, 0.90, 0.90),
                (220, 31, 0.04, 20, 8.0, 0.75, 0.80),
            ):
                specs.append(
                    make_lightgbm_spec(
                        n_estimators=params[0],
                        num_leaves=params[1],
                        learning_rate=params[2],
                        min_child_samples=params[3],
                        reg_lambda=params[4],
                        feature_fraction=params[5],
                        bagging_fraction=params[6],
                        weight_mode=weight_mode,
                        random_state=random_state,
                    )
                )
    except Exception:
        pass

    return specs


def fit_candidates(
    specs: list[CandidateSpec],
    train_frame: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
) -> list[FittedCandidate]:
    candidates: list[FittedCandidate] = []
    for index, spec in enumerate(specs, start=1):
        model = fit_spec(spec, x_train, y_train, train_frame)
        validation_scores = model_scores(model, x_validation)
        test_scores = model_scores(model, x_test)
        candidates.append(
            FittedCandidate(
                spec=spec,
                model=model,
                validation_scores=validation_scores,
                test_scores=test_scores,
                validation_metrics=evaluate_scores(validation_scores, y_validation),
                test_metrics=evaluate_scores(test_scores, y_test),
            )
        )
        if index % 20 == 0 or index == len(specs):
            print(f"fit {index}/{len(specs)} candidates", flush=True)
    return candidates


def candidate_sort_key(candidate: FittedCandidate) -> tuple[float, float]:
    return (
        float(candidate.validation_metrics["subnet_reward"]),
        float(candidate.validation_metrics["average_precision"]),
    )


def add_ensembles(
    candidates: list[FittedCandidate],
    y_validation: np.ndarray,
    y_test: np.ndarray,
) -> list[FittedCandidate]:
    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    ensembles: list[FittedCandidate] = []

    for top_n in (2, 3, 5, 8, 12):
        selected = ranked[:top_n]
        if len(selected) < top_n:
            continue

        for mode in ("mean", "validation_weighted"):
            if mode == "mean":
                weights = None
                validation_scores = np.mean([item.validation_scores for item in selected], axis=0)
                test_scores = np.mean([item.test_scores for item in selected], axis=0)
            else:
                raw_weights = np.asarray(
                    [float(item.validation_metrics["subnet_reward"]) for item in selected],
                    dtype=float,
                )
                raw_weights = raw_weights - raw_weights.min() + 0.001
                weights = raw_weights / raw_weights.sum()
                validation_scores = np.average(
                    [item.validation_scores for item in selected],
                    axis=0,
                    weights=weights,
                )
                test_scores = np.average([item.test_scores for item in selected], axis=0, weights=weights)

            model = ProbabilityAveragingEnsemble(
                [item.model for item in selected],
                weights=weights.tolist() if weights is not None else None,
            )
            spec = CandidateSpec(
                name=f"ensemble_top{top_n}_{mode}",
                model_type="probability_average_ensemble",
                params={"members": [item.spec.name for item in selected], "mode": mode},
                weight_mode="member_specific",
                factory=lambda: copy.deepcopy(model),
            )
            ensembles.append(
                FittedCandidate(
                    spec=spec,
                    model=model,
                    validation_scores=np.asarray(validation_scores, dtype=float),
                    test_scores=np.asarray(test_scores, dtype=float),
                    validation_metrics=evaluate_scores(np.asarray(validation_scores, dtype=float), y_validation),
                    test_metrics=evaluate_scores(np.asarray(test_scores, dtype=float), y_test),
                )
            )

    return candidates + ensembles


def serialize_candidate(candidate: FittedCandidate) -> dict[str, Any]:
    return {
        "name": candidate.spec.name,
        "model_type": candidate.spec.model_type,
        "params": candidate.spec.params,
        "weight_mode": candidate.spec.weight_mode,
        "validation": candidate.validation_metrics,
        "test": candidate.test_metrics,
    }


def refit_for_final(
    selected: FittedCandidate,
    selected_pool: list[FittedCandidate],
    final_frame: pd.DataFrame,
    x_final: pd.DataFrame,
    y_final: np.ndarray,
) -> Any:
    if selected.spec.model_type != "probability_average_ensemble":
        return fit_spec(selected.spec, x_final, y_final, final_frame)

    by_name = {candidate.spec.name: candidate for candidate in selected_pool}
    members = []
    for member_name in selected.spec.params["members"]:
        member = by_name[member_name]
        members.append(fit_spec(member.spec, x_final, y_final, final_frame))
    return ProbabilityAveragingEnsemble(members, weights=getattr(selected.model, "weights", None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, nargs="+", help="Training feature CSVs")
    parser.add_argument("--validation", required=True, help="Validation feature CSV")
    parser.add_argument("--test", required=True, help="Final test feature CSV")
    parser.add_argument("--output", default="models/poker44_tuned_selected.joblib")
    parser.add_argument("--final-output", default="models/poker44_tuned_final.joblib")
    parser.add_argument("--random-state", type=int, default=44)
    parser.add_argument("--max-candidates", type=int, default=0)
    args = parser.parse_args()

    train_paths = [Path(path) for path in args.train]
    train_frame = load_feature_csvs(train_paths)
    validation_frame = load_feature_csv(Path(args.validation))
    test_frame = load_feature_csv(Path(args.test))

    columns = feature_columns(train_frame)
    require_columns(validation_frame, columns, "Validation")
    require_columns(test_frame, columns, "Test")

    x_train = train_frame[columns].fillna(0.0)
    y_train = train_frame["label"].astype(int).to_numpy()
    x_validation = validation_frame[columns].fillna(0.0)
    y_validation = validation_frame["label"].astype(int).to_numpy()
    x_test = test_frame[columns].fillna(0.0)
    y_test = test_frame["label"].astype(int).to_numpy()

    specs = build_specs(args.random_state)
    if args.max_candidates > 0:
        specs = specs[: args.max_candidates]

    base_candidates = fit_candidates(
        specs,
        train_frame,
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    )
    candidates = add_ensembles(base_candidates, y_validation, y_test)
    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    selected = ranked[0]

    selected_bundle = {
        "model_name": selected.spec.name,
        "model": selected.model,
        "feature_columns": columns,
        "train_paths": [str(path) for path in train_paths],
        "validation_path": args.validation,
        "test_path": args.test,
        "selection_metric": "validation.subnet_reward",
        "selected": serialize_candidate(selected),
        "top_candidates": [serialize_candidate(candidate) for candidate in ranked[:15]],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_bundle, output_path)

    final_frame = pd.concat([train_frame, validation_frame], ignore_index=True, sort=False)
    x_final = final_frame[columns].fillna(0.0)
    y_final = final_frame["label"].astype(int).to_numpy()
    final_model = refit_for_final(selected, base_candidates, final_frame, x_final, y_final)
    final_scores = model_scores(final_model, x_test)
    final_metrics = evaluate_scores(final_scores, y_test)

    final_bundle = {
        "model_name": f"{selected.spec.name}_refit_through_validation",
        "model": final_model,
        "feature_columns": columns,
        "train_paths": [str(path) for path in train_paths] + [args.validation],
        "selection_train_paths": [str(path) for path in train_paths],
        "selection_validation_path": args.validation,
        "test_path": args.test,
        "selection_metric": "validation.subnet_reward",
        "selected": serialize_candidate(selected),
        "test": final_metrics,
    }
    final_output_path = Path(args.final_output)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_bundle, final_output_path)

    print(
        json.dumps(
            {
                "selected_output": str(output_path),
                "final_output": str(final_output_path),
                "feature_count": len(columns),
                "train_rows": int(len(train_frame)),
                "validation_rows": int(len(validation_frame)),
                "test_rows": int(len(test_frame)),
                "selected": serialize_candidate(selected),
                "final_refit_test": final_metrics,
                "top_candidates": [serialize_candidate(candidate) for candidate in ranked[:8]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
