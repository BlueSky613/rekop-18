"""Sweep Poker44 training start dates for a fixed model recipe."""

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

from poker44.modeling import ProbabilityAveragingEnsemble
from scripts.tune_model import (
    build_specs,
    evaluate_scores,
    feature_columns,
    fit_spec,
    load_feature_csv,
    load_feature_csvs,
    model_scores,
    require_columns,
)


DEFAULT_MEMBER_NAMES = [
    "hgb_lr0.05_iter350_leaf7_l20.05_min10_full_release_boost",
    "hgb_lr0.05_iter200_leaf7_l20.05_min10_full_release_boost",
    "extra_trees_n700_min2_feat0.5_none",
    "hgb_lr0.03_iter350_leaf15_l20.0_min10_recent_soft",
    "hgb_lr0.05_iter350_leaf7_l20.0_min10_recent_soft",
]


def available_dates(data_dir: Path) -> list[str]:
    dates = []
    for child in data_dir.iterdir():
        if child.is_dir() and (child / "features.csv").exists():
            dates.append(child.name)
    return sorted(dates)


def csv_paths(data_dir: Path, dates: list[str]) -> list[Path]:
    return [data_dir / date / "features.csv" for date in dates]


def load_member_names(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_MEMBER_NAMES
    bundle = joblib.load(path)
    selected = bundle.get("selected") or {}
    params = selected.get("params") or {}
    members = params.get("members")
    if not members:
        return DEFAULT_MEMBER_NAMES
    return [str(member) for member in members]


def fit_member_scores(
    *,
    member_names: list[str],
    train_frame: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
    specs_by_name: dict[str, Any],
) -> np.ndarray:
    scores = []
    for member_name in member_names:
        spec = specs_by_name[member_name]
        model = fit_spec(spec, x_train, y_train, train_frame)
        scores.append(model_scores(model, x_eval))
    return np.mean(scores, axis=0)


def fit_final_model(
    *,
    member_names: list[str],
    train_frame: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    specs_by_name: dict[str, Any],
) -> ProbabilityAveragingEnsemble:
    models = []
    for member_name in member_names:
        spec = specs_by_name[member_name]
        models.append(fit_spec(spec, x_train, y_train, train_frame))
    return ProbabilityAveragingEnsemble(models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="downloads/poker44_benchmark")
    parser.add_argument("--min-start", default="2026-05-26")
    parser.add_argument("--selection-train-end", default="2026-07-08")
    parser.add_argument("--validation-date", default="2026-07-09")
    parser.add_argument("--final-train-end", default="2026-07-09")
    parser.add_argument("--test-date", default="2026-07-10")
    parser.add_argument("--model-bundle", default="models/poker44_204f_tuned_final_through_2026-07-09_test_2026-07-10.joblib")
    parser.add_argument("--output", default="models/start_date_sweep.json")
    parser.add_argument("--best-output", default="")
    parser.add_argument("--random-state", type=int, default=44)
    parser.add_argument(
        "--starts",
        nargs="*",
        default=[],
        help="Optional explicit start dates. Defaults to every available date in range.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dates = available_dates(data_dir)
    starts = args.starts or [
        date
        for date in dates
        if args.min_start <= date <= args.selection_train_end
    ]

    validation_frame = load_feature_csv(data_dir / args.validation_date / "features.csv")
    test_frame = load_feature_csv(data_dir / args.test_date / "features.csv")
    member_names = load_member_names(Path(args.model_bundle) if args.model_bundle else None)
    specs_by_name = {spec.name: spec for spec in build_specs(args.random_state)}
    missing_members = sorted(set(member_names) - set(specs_by_name))
    if missing_members:
        raise RuntimeError(f"Missing model specs for members: {missing_members}")

    results: list[dict[str, Any]] = []
    for start in starts:
        selection_dates = [
            date for date in dates if start <= date <= args.selection_train_end
        ]
        final_dates = [date for date in dates if start <= date <= args.final_train_end]
        if not selection_dates or not final_dates:
            continue

        selection_frame = load_feature_csvs(csv_paths(data_dir, selection_dates))
        columns = feature_columns(selection_frame)
        require_columns(validation_frame, columns, "Validation")
        require_columns(test_frame, columns, "Test")

        x_selection = selection_frame[columns].fillna(0.0)
        y_selection = selection_frame["label"].astype(int).to_numpy()
        x_validation = validation_frame[columns].fillna(0.0)
        y_validation = validation_frame["label"].astype(int).to_numpy()
        x_test = test_frame[columns].fillna(0.0)
        y_test = test_frame["label"].astype(int).to_numpy()

        validation_scores = fit_member_scores(
            member_names=member_names,
            train_frame=selection_frame,
            x_train=x_selection,
            y_train=y_selection,
            x_eval=x_validation,
            specs_by_name=specs_by_name,
        )
        validation_metrics = evaluate_scores(validation_scores, y_validation)

        final_frame = load_feature_csvs(csv_paths(data_dir, final_dates))
        x_final = final_frame[columns].fillna(0.0)
        y_final = final_frame["label"].astype(int).to_numpy()
        final_model = fit_final_model(
            member_names=member_names,
            train_frame=final_frame,
            x_train=x_final,
            y_train=y_final,
            specs_by_name=specs_by_name,
        )
        test_scores = model_scores(final_model, x_test)
        test_metrics = evaluate_scores(test_scores, y_test)

        row = {
            "start": start,
            "selection_train_dates": selection_dates,
            "final_train_dates": final_dates,
            "selection_train_rows": int(len(selection_frame)),
            "final_train_rows": int(len(final_frame)),
            "feature_count": int(len(columns)),
            "member_names": member_names,
            "validation": validation_metrics,
            "test": test_metrics,
        }
        results.append(row)
        print(
            json.dumps(
                {
                    "start": start,
                    "validation_reward": validation_metrics["subnet_reward"],
                    "test_reward": test_metrics["subnet_reward"],
                    "test_ap": test_metrics["average_precision"],
                    "test_recall": test_metrics["bot_recall"],
                    "test_fpr": test_metrics["fpr"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    by_validation = sorted(
        results,
        key=lambda row: (
            float(row["validation"]["subnet_reward"]),
            float(row["validation"]["average_precision"]),
        ),
        reverse=True,
    )
    by_test = sorted(
        results,
        key=lambda row: (
            float(row["test"]["subnet_reward"]),
            float(row["test"]["average_precision"]),
        ),
        reverse=True,
    )
    payload = {
        "member_names": member_names,
        "top_by_validation": by_validation[:10],
        "top_by_test": by_test[:10],
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.best_output and by_validation:
        best = by_validation[0]
        final_dates = best["final_train_dates"]
        final_frame = load_feature_csvs(csv_paths(data_dir, final_dates))
        columns = feature_columns(final_frame)
        x_final = final_frame[columns].fillna(0.0)
        y_final = final_frame["label"].astype(int).to_numpy()
        best_model = fit_final_model(
            member_names=member_names,
            train_frame=final_frame,
            x_train=x_final,
            y_train=y_final,
            specs_by_name=specs_by_name,
        )
        best_bundle = {
            "model_name": "start_sweep_validation_selected_ensemble",
            "model": best_model,
            "feature_columns": columns,
            "train_paths": [str(path) for path in csv_paths(data_dir, final_dates)],
            "selection_metric": "validation.subnet_reward",
            "selected_start": best["start"],
            "selected": best,
            "sweep_output": str(output_path),
        }
        best_output_path = Path(args.best_output)
        best_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_bundle, best_output_path)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "best_output": args.best_output or None,
                "best_by_validation": by_validation[0] if by_validation else None,
                "best_by_test": by_test[0] if by_test else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
