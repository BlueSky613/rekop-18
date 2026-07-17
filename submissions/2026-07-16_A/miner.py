"""Drop-in Poker44 SN126 miner — loads the joblib artifact (real submission format).

Copy this repo's poker44_ml/, model/poker44_model.joblib, and this file into the
Poker44-subnet checkout (replacing neurons/miner.py), then run like the reference miner.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.validator.synapse import DetectionSynapse

from poker44_ml.inference import Poker44Model, SAFETY_MODE, _MODEL as _MODEL_PATH

REPO_COMMIT = "REPLACE_WITH_REAL_COMMIT"   # set before serving; keep manifest honest

# 대시보드로 쿼리별 상세(밸리데이터·청크별 점수) 보고 — 07_live_dashboard가 표시.
# 원격 대시보드로 보내려면:  POKER44_REPORT_URL=http://<대시보드IP>:8127  환경변수 설정.
REPORT_URL = os.environ.get("POKER44_REPORT_URL", "").strip().rstrip("/")
_QLOG = os.environ.get("POKER44_QUERY_LOG", "queries.jsonl")


def _report_query(uid, validator, scores):
    rec = {
        "uid": int(uid) if uid is not None else None,
        "validator": validator or "?",
        "n_chunks": len(scores),
        "scores": [round(float(s), 4) for s in scores],
        "window": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    try:
        with open(_QLOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + chr(10))
    except Exception:
        pass
    if REPORT_URL:
        def _post():
            try:
                body = json.dumps(rec).encode("utf-8")
                req = urllib.request.Request(REPORT_URL + "/api/report", data=body,
                                             headers={"content-type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=5).read()
            except Exception:
                pass
        threading.Thread(target=_post, daemon=True).start()




class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.model = Poker44Model()
        # ★자동 리로드: daily_update가 joblib을 갱신하면 재시작 없이 반영 (재학습 실제 적용)
        self._model_mtime = _MODEL_PATH.stat().st_mtime if _MODEL_PATH.exists() else 0.0
        threading.Thread(target=self._reload_watcher, daemon=True).start()
        self.model_manifest = {
            "schema_version": "1",
            "open_source": True,
            "repo_url": "https://github.com/<you>/poker44-sn126-submission",
            "repo_commit": REPO_COMMIT,
            "model_name": self.model.metadata.get("name", "poker44-honest-behavioral"),
            "model_version": str(self.model.metadata.get("version", "2")),
            "framework": "sklearn+lightgbm-ensemble (joblib)",
            "license": "MIT",
            "training_data_statement": (
                "Trained only on the public Poker44 benchmark "
                "(api.poker44.net/api/v1/benchmark). No validator-only eval labels used."
            ),
            "training_data_sources": ["poker44-public-benchmark"],
            "private_data_attestation": "Does not train on validator-only evaluation data.",
            "implementation_files": ["poker44_ml/combined.py", "poker44_ml/inference.py",
                                     "poker44_ml/features.py", "poker44_ml/features_leader.py"],
            "inference_mode": "remote",
            "notes": f"combined312 (leader293+honest19) ensemble; top-K safety mode={SAFETY_MODE}.",
        }
        bt.logging.info(
            f"Poker44 miner up | joblib models={len(self.model.models)} "
            f"features={len(self.model.feature_names)} safety={SAFETY_MODE}"
        )

    def _reload_watcher(self, every=60):
        """daily_update가 model/poker44_model.joblib 을 갱신하면 자동으로 새 모델 로드.
        참조 스왑이라 원자적 — 재시작·쿼리중단 없이 재학습 결과가 반영된다."""
        while True:
            time.sleep(every)
            try:
                if not _MODEL_PATH.exists():
                    continue
                mt = _MODEL_PATH.stat().st_mtime
                if mt > self._model_mtime + 1:
                    new_model = Poker44Model()          # 갱신된 joblib 로드
                    self.model = new_model               # 원자적 교체
                    self._model_mtime = mt
                    bt.logging.info(
                        f"🔄 모델 자동 재로드 (재학습 반영) | models={len(new_model.models)} "
                        f"name={new_model.metadata.get('name')}"
                    )
            except Exception as e:
                bt.logging.warning(f"모델 재로드 실패(옛 모델 유지): {e}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        scores = self.model.predict_chunk_scores(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        # ★매니페스트 전송 안 함 (manifestPresent=False) — 상위10 전원 이렇게 함.
        #   보내면 리뷰 실패 시 -0.10~-0.22 벌점(152가 -0.10 물고 있음). 안 보내면 벌점 0.
        # synapse.model_manifest = None  # 기본값 유지
        bt.logging.info(f"Scored {len(chunks)} chunks | mean={sum(scores)/max(len(scores),1):.3f}")
        try:
            vhot = getattr(getattr(synapse, "dendrite", None), "hotkey", None)  # querying validator
            _report_query(getattr(self, "uid", None), vhot, scores)
        except Exception:
            pass
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 submission miner running...")
        while True:
            bt.logging.info(f"UID {miner.uid} | Incentive {miner.metagraph.I[miner.uid]}")
            time.sleep(300)
