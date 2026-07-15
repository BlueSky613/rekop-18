"""Poker44 miner wired to local 127/08_submission joblib artifacts."""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse
from poker44_ml.inference import Poker44Model, SAFETY_MODE


DEFAULT_MODEL_VERSION = "127-submission"


class Miner(BaseMinerNeuron):
    """Local SN126 miner serving the 127 submission's batch scorer."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        print("[STARTUP] Poker44 127-submission miner started", flush=True)
        bt.logging.info("Poker44 127-submission miner started")

        repo_root = Path(__file__).resolve().parents[1]
        self.model_path = self._resolve_model_path(repo_root)
        self.request_log_path = self._resolve_request_log_path(repo_root)
        self.model = self._load_submission_model(self.model_path)
        metadata = dict(getattr(self.model, "metadata", {}) or {})

        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                repo_root / "poker44_ml" / "inference.py",
                repo_root / "poker44_ml" / "combined.py",
                repo_root / "poker44_ml" / "features.py",
                repo_root / "poker44_ml" / "features_leader.py",
            ],
            defaults={
                "model_name": metadata.get("name", "poker44-honest-behavioral"),
                "model_version": str(metadata.get("version", DEFAULT_MODEL_VERSION)),
                "framework": "lightgbm+scikit-learn-joblib",
                "license": "MIT",
                "artifact_sha256": self._sha256_file(self.model_path),
                "notes": (
                    "127/08_submission local joblib ensemble. "
                    f"Artifact version={metadata.get('version', '2')}; "
                    f"strategy={metadata.get('strategy', 'unspecified')}; "
                    f"Safety mode={SAFETY_MODE}; feature_set={metadata.get('feature_set', 'combined312')}; "
                    f"models={len(self.model.models)}; features={len(self.model.feature_names)}."
                ),
                "open_source": True,
                "inference_mode": "local",
                "training_data_statement": (
                    "Trained only on the public Poker44 benchmark data available through "
                    "api.poker44.net/api/v1/benchmark. No validator-only eval labels used."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        if self.request_log_path is not None:
            self.request_log_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"[STARTUP] request summaries path={self.request_log_path}",
                flush=True,
            )

        bt.logging.info(f"Axon created: {self.axon}")

    def _resolve_model_path(self, repo_root: Path) -> Path:
        configured = str(getattr(self.config.miner, "model_path", "") or "").strip()
        path = Path(configured) if configured else Path()
        if not path.is_absolute():
            path = repo_root / path
        return path

    def _resolve_request_log_path(self, repo_root: Path) -> Path | None:
        configured = os.getenv(
            "POKER44_REQUEST_LOG_PATH",
            "logs/forward_requests.jsonl",
        ).strip()
        if not configured:
            return None
        path = Path(configured)
        if not path.is_absolute():
            path = repo_root / path
        return path

    def _load_submission_model(self, model_path: Path) -> Poker44Model:
        if not model_path.exists():
            raise FileNotFoundError(f"127 submission model artifact not found: {model_path}")

        try:
            model = Poker44Model(model_path)
        except Exception as exc:
            bt.logging.error(f"Failed to load 127 submission model {model_path}: {exc}")
            raise RuntimeError(f"Failed to load 127 submission model {model_path}") from exc

        if not model.models or not model.feature_names:
            raise ValueError(f"Invalid 127 submission model artifact: {model_path}")

        print(
            f"[MODEL] loaded 127 submission path={model_path} "
            f"name={model.metadata.get('name', '')} version={model.metadata.get('version', '')} "
            f"models={len(model.models)} features={len(model.feature_names)} "
            f"safety={SAFETY_MODE}",
            flush=True,
        )
        bt.logging.info(
            f"Loaded 127 submission model {model_path} | "
            f"models={len(model.models)} features={len(model.feature_names)} safety={SAFETY_MODE}"
        )
        return model

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _log_manifest_startup(self, repo_root: Path) -> None:
        print(
            "[MODEL] manifest "
            f"name={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"digest={self.manifest_digest}",
            flush=True,
        )
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    @staticmethod
    def _json_sha256(payload) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _first_hand_schema(chunks) -> dict:
        if not chunks:
            return {}
        first_chunk = chunks[0] if isinstance(chunks[0], list) else []
        if not first_chunk or not isinstance(first_chunk[0], dict):
            return {}
        first_hand = first_chunk[0]
        metadata = first_hand.get("metadata") or {}
        return {
            "top_level_keys": sorted(str(key) for key in first_hand.keys()),
            "metadata_keys": (
                sorted(str(key) for key in metadata.keys())
                if isinstance(metadata, dict)
                else []
            ),
        }

    def _write_request_summary(self, synapse: DetectionSynapse, chunks, scores) -> None:
        if self.request_log_path is None:
            return

        dendrite = getattr(synapse, "dendrite", None)
        chunk_sizes = [len(chunk) if isinstance(chunk, list) else 0 for chunk in chunks]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        record = {
            "timestamp_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "uid": int(self.uid) if self.uid is not None else None,
            "block": int(self.block) if self.block is not None else None,
            "caller_hotkey": getattr(dendrite, "hotkey", None),
            "caller_uuid": getattr(dendrite, "uuid", None),
            "chunk_count": len(chunks),
            "hand_count": int(sum(chunk_sizes)),
            "chunk_sizes_first20": chunk_sizes[:20],
            "chunk_sizes_truncated": len(chunk_sizes) > 20,
            "chunks_sha256": self._json_sha256(chunks),
            "first_hand_schema": self._first_hand_schema(chunks),
            "score_count": len(scores),
            "score_mean": round(float(mean_score), 8),
            "first_scores": [round(float(score), 6) for score in scores[:20]],
            "model_path": str(self.model_path),
            "model_name": self.model_manifest.get("model_name", ""),
            "model_version": self.model_manifest.get("model_version", ""),
            "manifest_digest": self.manifest_digest,
            "safety_mode": SAFETY_MODE,
        }
        with self.request_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one bot-risk score per chunk using the submission batch scorer."""
        chunks = synapse.chunks or []
        print(f"[FORWARD] received chunks={len(chunks)}", flush=True)

        try:
            scores = self.model.predict_chunk_scores(chunks)
            if len(scores) != len(chunks):
                raise ValueError(
                    f"model returned {len(scores)} scores for {len(chunks)} chunks"
                )
        except Exception as exc:
            print(
                f"[FORWARD] batch_score_failed error={exc}; using neutral scores",
                flush=True,
            )
            bt.logging.error(f"Batch scoring failed; using neutral scores: {exc}")
            scores = [0.5 for _ in chunks]

        synapse.risk_scores = scores
        synapse.predictions = [score >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        try:
            self._write_request_summary(synapse, chunks, scores)
        except Exception as exc:
            print(f"[REQUEST_LOG] failed error={exc}", flush=True)
            bt.logging.warning(f"Request summary logging failed: {exc}")
        print(
            f"[FORWARD] scored chunks={len(chunks)} scores={len(scores)} "
            f"mean={mean_score:.4f} first_scores={[round(score, 4) for score in scores[:20]]}",
            flush=True,
        )
        bt.logging.info(f"Miner predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks.")
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        print("[STARTUP] Poker44 miner running", flush=True)
        bt.logging.info("Poker44 miner running...")
        while True:
            print(
                f"[HEARTBEAT] uid={miner.uid} block={miner.block} "
                f"incentive={miner.metagraph.I[miner.uid]}",
                flush=True,
            )
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
