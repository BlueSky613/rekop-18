"""Build chunk-level feature CSVs from downloaded Poker44 benchmark JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.miner_model import extract_chunk_features


def iter_examples(chunks_jsonl: Path):
    with chunks_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            groups = item.get("chunks")
            labels = item.get("groundTruth")
            label_names = item.get("groundTruthLabels") or []
            if not isinstance(groups, list) or not isinstance(labels, list):
                raise RuntimeError(f"Invalid benchmark item at line {line_number}")
            if len(groups) != len(labels):
                raise RuntimeError(
                    f"Group/label mismatch at line {line_number}: {len(groups)} != {len(labels)}"
                )

            for group_index, (group, label) in enumerate(zip(groups, labels)):
                if not isinstance(group, list):
                    continue
                label_int = int(label)
                yield {
                    "source_date": item.get("sourceDate", ""),
                    "split": item.get("split", ""),
                    "chunk_id": item.get("chunkId", ""),
                    "chunk_hash": item.get("chunkHash", ""),
                    "chunk_index": item.get("chunkIndex", ""),
                    "group_index": group_index,
                    "label": label_int,
                    "label_name": (
                        str(label_names[group_index])
                        if isinstance(label_names, list) and group_index < len(label_names)
                        else ("bot" if label_int == 1 else "human")
                    ),
                    "group": group,
                }


def build_feature_rows(chunks_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in iter_examples(chunks_jsonl):
        group = example.pop("group")
        features = extract_chunk_features(group)
        rows.append({**example, **features})
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise RuntimeError("No feature rows to write")

    metadata_columns = [
        "source_date",
        "split",
        "chunk_id",
        "chunk_hash",
        "chunk_index",
        "group_index",
        "label",
        "label_name",
    ]
    feature_columns = sorted(
        key for key in rows[0].keys() if key not in set(metadata_columns)
    )
    fieldnames = metadata_columns + feature_columns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="downloads/poker44_benchmark/2026-07-09/chunks.jsonl",
        help="Downloaded chunks.jsonl path",
    )
    parser.add_argument(
        "--output",
        default="downloads/poker44_benchmark/2026-07-09/features.csv",
        help="Feature CSV output path",
    )
    args = parser.parse_args()

    rows = build_feature_rows(Path(args.input))
    write_csv(rows, Path(args.output))

    label_counts: dict[int, int] = {}
    for row in rows:
        label = int(row["label"])
        label_counts[label] = label_counts.get(label, 0) + 1

    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "rows": len(rows),
                "labelCounts": {
                    "human": label_counts.get(0, 0),
                    "bot": label_counts.get(1, 0),
                },
                "featureCount": len(rows[0]) - 8 if rows else 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
