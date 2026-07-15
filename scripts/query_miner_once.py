"""Send one signed Poker44 DetectionSynapse to a miner axon.

This is a protocol-level smoke test. With strict miner blacklist settings, a
non-validator test wallet may be rejected before forward(), but the miner log
should still show a [REQUEST] line if the signed request reached the miner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import bittensor as bt

from poker44.validator.synapse import DetectionSynapse


def load_chunks(path: Path, limit: int) -> list[list[dict[str, Any]]]:
    """Load a few benchmark chunks from jsonl, with a tiny fallback chunk."""
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                groups = row.get("chunks") or []
                chunks: list[list[dict[str, Any]]] = []
                for group in groups:
                    if isinstance(group, list):
                        chunks.append(group)
                    elif isinstance(group, dict):
                        hands = group.get("hands")
                        if isinstance(hands, list):
                            chunks.append(hands)
                if chunks:
                    return chunks[:limit]

    return [
        [
            {
                "players": [{"seat": 1}, {"seat": 2}],
                "actions": [
                    {"action_type": "call"},
                    {"action_type": "check"},
                    {"action_type": "fold"},
                ],
                "streets": ["preflop"],
                "outcome": {"showdown": False},
            }
        ]
    ]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=126)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--wallet.name", dest="wallet_name", required=True)
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", required=True)
    parser.add_argument(
        "--chunks-jsonl",
        default="downloads/poker44_benchmark/2026-07-12/chunks.jsonl",
    )
    parser.add_argument("--chunk-count", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    subtensor = bt.Subtensor(network=args.network)
    metagraph = subtensor.metagraph(args.netuid)
    axon = metagraph.axons[args.uid]
    chunks = load_chunks(Path(args.chunks_jsonl), args.chunk_count)
    synapse = DetectionSynapse(chunks=chunks)

    print(f"[SELFTEST] target_uid={args.uid} axon={axon}")
    print(f"[SELFTEST] sender_hotkey={wallet.hotkey.ss58_address}")
    print(f"[SELFTEST] sending chunks={len(chunks)} timeout={args.timeout}")

    dendrite = bt.Dendrite(wallet=wallet)
    responses = await dendrite(axons=[axon], synapse=synapse, timeout=args.timeout)
    response = responses[0] if responses else None

    print(f"[SELFTEST] response_type={type(response).__name__}")
    if response is None:
        print("[SELFTEST] no response")
        return

    print(f"[SELFTEST] risk_scores={getattr(response, 'risk_scores', None)}")
    print(f"[SELFTEST] predictions={getattr(response, 'predictions', None)}")
    dendrite_info = getattr(response, "dendrite", None)
    print(f"[SELFTEST] dendrite={dendrite_info}")


if __name__ == "__main__":
    asyncio.run(main())
