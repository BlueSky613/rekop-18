"""Download Poker44 public benchmark releases for local miner training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://api.poker44.net/api/v1/benchmark"


def _get_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success", False):
        raise RuntimeError(f"API returned unsuccessful response for {response.url}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected API data shape for {response.url}")
    return data


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def download_source_date(source_date: str, out_dir: Path, *, limit: int) -> dict[str, Any]:
    source_dir = out_dir / source_date
    chunks_path = source_dir / "chunks.jsonl"
    if chunks_path.exists():
        chunks_path.unlink()

    cursor = None
    page_count = 0
    item_count = 0
    group_count = 0
    hand_count = 0
    label_counts = {"human": 0, "bot": 0}

    while True:
        params: dict[str, Any] = {"sourceDate": source_date, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        page = _get_json(f"{BASE_URL}/chunks", params=params)
        rows = page.get("chunks", [])
        if not isinstance(rows, list):
            raise RuntimeError("Unexpected chunks page shape")

        _append_jsonl(chunks_path, rows)
        page_count += 1
        item_count += len(rows)

        for item in rows:
            hand_count += int(item.get("handCount") or 0)
            labels = item.get("groundTruth") or []
            groups = item.get("chunks") or []
            group_count += len(groups) if isinstance(groups, list) else 0
            for label in labels if isinstance(labels, list) else []:
                if int(label) == 1:
                    label_counts["bot"] += 1
                else:
                    label_counts["human"] += 1

        cursor = page.get("nextCursor")
        if not cursor:
            break

    summary = {
        "sourceDate": source_date,
        "pages": page_count,
        "benchmarkItems": item_count,
        "chunkGroups": group_count,
        "hands": hand_count,
        "labelCounts": label_counts,
        "chunksJsonl": str(chunks_path),
    }
    _write_json(source_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-date", default="latest", help="YYYY-MM-DD or latest")
    parser.add_argument("--out", default="downloads/poker44_benchmark")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--quiet", action="store_true", help="Print only the download summary")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    status = _get_json(BASE_URL)
    releases = _get_json(f"{BASE_URL}/releases", params={"limit": 50})
    _write_json(out_dir / "status.json", status)
    _write_json(out_dir / "releases.json", releases)

    source_date = args.source_date
    if source_date == "latest":
        source_date = str(status["latestSourceDate"])

    summary = download_source_date(source_date, out_dir, limit=max(1, args.limit))
    payload = summary if args.quiet else {"status": status, "download": summary}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
