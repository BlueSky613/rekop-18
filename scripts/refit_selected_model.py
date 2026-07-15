"""Refit an existing Poker44 selected recipe through a new training end date."""

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
from scripts.tune_model import build_specs, fit_spec, load_feature_csvs, require_columns


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


def selected_recipe_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    recipe = bundle.get("selected_recipe")
    if isinstance(recipe, dict) and recipe.get("members"):
        return recipe

    selected = bundle.get("selected")
    if isinstance(selected, dict):
        params = selected.get("params")
        if isinstance(params, dict) and params.get("members"):
            return {
                "name": selected.get("name", bundle.get("model_name", "selected_recipe")),
                "members": list(params["members"]),
                "weights": getattr(bundle.get("model"), "weights", None),
            }

    raise RuntimeError("Source model does not contain a selected recipe with members")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model",
        required=True,
        help="Existing selected model bundle to copy the recipe from",
    )
    parser.add_argument("--data-dir", default="downloads/poker44_benchmark")
    parser.add_argument("--train-start", default="2026-06-25")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-state", type=int, default=44)
    args = parser.parse_args()

    source_path = Path(args.source_model)
    source_bundle = joblib.load(source_path)
    recipe = selected_recipe_from_bundle(source_bundle)
    member_names = [str(name) for name in recipe["members"]]
    weights = recipe.get("weights")
    if weights is not None:
        weights = [float(weight) for weight in weights]

    specs_by_name = {spec.name: spec for spec in build_specs(args.random_state)}
    missing = sorted(set(member_names) - set(specs_by_name))
    if missing:
        raise RuntimeError(f"Selected recipe references unknown specs: {missing}")

    paths = feature_paths(Path(args.data_dir), args.train_start, args.train_end)
    frame = load_feature_csvs(paths)
    columns = list(source_bundle.get("feature_columns") or [])
    if not columns:
        raise RuntimeError("Source model bundle does not contain feature_columns")
    require_columns(frame, columns, "Training")

    x_train = frame[columns].fillna(0.0)
    y_train = frame["label"].astype(int).to_numpy()

    estimators = []
    fitted_members = []
    for member_name in member_names:
        spec = specs_by_name[member_name]
        estimator = fit_spec(spec, x_train, y_train, frame)
        estimators.append(estimator)
        fitted_members.append(
            {
                "name": spec.name,
                "model_type": spec.model_type,
                "params": spec.params,
                "weight_mode": spec.weight_mode,
            }
        )

    model = ProbabilityAveragingEnsemble(estimators, weights=weights)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_name": f"{recipe.get('name', 'selected_recipe')}_refit_through_{args.train_end}",
        "model": model,
        "feature_columns": columns,
        "train_paths": [str(path) for path in paths],
        "selection_metric": source_bundle.get("selection_metric", "validation.subnet_reward"),
        "selected_recipe": {
            **recipe,
            "members": member_names,
            "weights": weights,
            "refit_source_model": str(source_path),
            "refit_train_start": args.train_start,
            "refit_train_end": args.train_end,
        },
        "fitted_members": fitted_members,
        "notes": (
            f"Same selected recipe as {source_path.name}; refit on public benchmark "
            f"features from {args.train_start} through {args.train_end}."
        ),
    }
    joblib.dump(bundle, output_path)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "source_model": str(source_path),
                "model_name": bundle["model_name"],
                "feature_count": len(columns),
                "train_start": args.train_start,
                "train_end": args.train_end,
                "train_rows": int(len(frame)),
                "label_counts": {
                    "human": int(np.sum(y_train == 0)),
                    "bot": int(np.sum(y_train == 1)),
                },
                "members": member_names,
                "weights": weights,
                "train_paths": [str(path) for path in paths],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
