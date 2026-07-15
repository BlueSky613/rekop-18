"""Validate a Poker44 submission joblib model on a downloaded benchmark date."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44_ml.inference import Poker44Model, SAFETY_MODE


def recall_at_fpr(scores: list[float], labels: list[int], max_fpr: float = 0.05) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if not pos or not neg:
        return 0.0

    order = np.argsort(-s, kind="mergesort")
    sorted_labels = y[order]
    recall = np.cumsum(sorted_labels == 1) / pos
    fpr = np.cumsum(sorted_labels == 0) / neg
    allowed = fpr <= max_fpr
    return float(recall[allowed].max()) if allowed.any() else 0.0


def reward(scores: list[float], labels: list[int]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())

    ap = float(average_precision_score(y, s)) if pos else 0.0
    auc = float(roc_auc_score(y, s)) if pos and neg else 0.0
    r5 = recall_at_fpr(scores, labels)

    hard = s >= 0.5
    tp = int((hard & (y == 1)).sum())
    fp = int((hard & (y == 0)).sum())
    hard_fpr = fp / max(neg, 1)

    if not pos or not neg:
        safety = 1.0
    elif tp == 0:
        safety = 0.0
    elif hard_fpr <= 0.1:
        safety = 1.0
    else:
        safety = max(0.0, 1.0 - (hard_fpr - 0.1) / 0.9)

    composite = (
        0.0
        if safety <= 0
        else float(np.clip(0.35 * ap + 0.30 * r5 + 0.20 * safety + 0.10 * safety + 0.05, 0, 1))
    )
    return {
        "composite": composite,
        "ap": ap,
        "auc": auc,
        "recall_at_5pct_fpr": r5,
        "safety": float(safety),
        "hard_fpr": float(hard_fpr),
        "tp": tp,
        "fp": fp,
        "pos": pos,
        "neg": neg,
    }


def iter_benchmark_items(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            chunks = item.get("chunks") or []
            labels = [int(value) for value in (item.get("groundTruth") or [])]
            if len(chunks) != len(labels):
                raise RuntimeError(
                    f"chunk/label mismatch at line {line_number}: {len(chunks)} != {len(labels)}"
                )
            yield line_number, item, chunks, labels


def validate(model_path: Path, chunks_jsonl: Path, *, limit_rows: int | None = None) -> dict[str, Any]:
    model = Poker44Model(model_path)
    all_scores: list[float] = []
    all_labels: list[int] = []
    batch_metrics: list[dict[str, Any]] = []
    rows = 0

    for line_number, item, chunks, labels in iter_benchmark_items(chunks_jsonl):
        if limit_rows is not None and rows >= limit_rows:
            break
        rows += 1
        scores = model.predict_chunk_scores(chunks)
        if len(scores) != len(labels):
            raise RuntimeError(
                f"score/label mismatch at line {line_number}: {len(scores)} != {len(labels)}"
            )
        all_scores.extend(scores)
        all_labels.extend(labels)
        batch_metrics.append(reward(scores, labels))

    if not all_labels:
        raise RuntimeError(f"No benchmark rows found in {chunks_jsonl}")

    labels_array = np.asarray(all_labels, dtype=int)
    return {
        "model": str(model_path),
        "chunks_jsonl": str(chunks_jsonl),
        "safety_mode": SAFETY_MODE,
        "rows": rows,
        "examples": len(all_labels),
        "label_counts": {
            "human": int((labels_array == 0).sum()),
            "bot": int((labels_array == 1).sum()),
        },
        "aggregate": reward(all_scores, all_labels),
        "batch_mean_composite": float(np.mean([m["composite"] for m in batch_metrics])),
        "batch_min_composite": float(np.min([m["composite"] for m in batch_metrics])),
        "batch_zero_composite_count": int(sum(1 for m in batch_metrics if m["composite"] <= 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-date", default="2026-07-14")
    parser.add_argument("--model-path", default="models/poker44_start_model_1.joblib")
    parser.add_argument("--input", default="")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    chunks_jsonl = (
        Path(args.input)
        if args.input
        else Path("downloads") / "poker44_benchmark" / args.source_date / "chunks.jsonl"
    )
    result = validate(
        Path(args.model_path),
        chunks_jsonl,
        limit_rows=args.limit_rows if args.limit_rows > 0 else None,
    )
    result["source_date"] = args.source_date

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
