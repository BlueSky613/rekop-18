"""Inference wrapper: load the Poker44 joblib artifact and score chunks.

Interface matches the live subnet: Poker44Model(path).predict_chunk_scores(chunks).
Blend = weighted mean of each model's P(bot). Safety = 152-proof top-K.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

from poker44_ml.combined import chunk_features

warnings.filterwarnings("ignore", message="X does not have valid feature names")

_MODEL = Path(__file__).resolve().parent.parent / "model" / "poker44_model.joblib"
SAFETY_MODE = os.environ.get("POKER44_SAFETY_MODE", "honest").strip().lower()


def _install_sklearn_pickle_compat():
    """Expose sklearn 1.7 loss symbols when newer sklearn moved them."""
    try:
        import sklearn._loss as sklearn_loss
        import sklearn._loss.loss as sklearn_loss_module
    except Exception:
        return

    for name in dir(sklearn_loss_module):
        if name.startswith("Cy") and not hasattr(sklearn_loss, name):
            setattr(sklearn_loss, name, getattr(sklearn_loss_module, name))
    sys.modules.setdefault("_loss", sklearn_loss)


class Poker44Model:
    def __init__(self, model_path=_MODEL):
        _install_sklearn_pickle_compat()
        art = joblib.load(model_path)
        self.models = list(art.get("models") or ([art["model"]] if art.get("model") else []))
        self.feature_names = list(art.get("feature_names") or [])
        weights = art.get("model_weights") or [1.0] * len(self.models)
        self.weights = np.asarray(weights[: len(self.models)], dtype=np.float64)
        if self.weights.sum() <= 0:
            self.weights = np.ones(len(self.models))
        self.weights /= self.weights.sum()
        self.metadata = dict(art.get("metadata") or {})

    def _rows(self, chunks):
        return np.array(
            [
                [chunk_features(chunk).get(name, 0.0) for name in self.feature_names]
                for chunk in chunks
            ],
            dtype=np.float64,
        )

    def _blend(self, x):
        preds = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                preds.append(np.clip(model.predict_proba(x)[:, 1], 0, 1))
            else:
                preds.append(np.clip(model.predict(x), 0, 1))
        return np.average(np.vstack(preds), axis=0, weights=self.weights)

    def _safe_topk(self, probabilities, mode):
        """Flag exactly K top-ranked chunks positive while preserving rank order."""
        n = len(probabilities)
        if n == 0:
            return probabilities

        k = max(1, int(math.floor(0.10 * n)))
        order = np.argsort(-probabilities, kind="mergesort")
        if mode == "band":
            positive_hi, positive_lo, negative_hi, negative_lo = 0.509, 0.501, 0.490, 0.010
        else:
            positive_hi, positive_lo, negative_hi, negative_lo = 0.900, 0.550, 0.450, 0.020

        out = np.empty(n, dtype=np.float64)
        for rank, idx in enumerate(order[:k]):
            out[idx] = positive_hi - (rank / max(k - 1, 1)) * (positive_hi - positive_lo)

        rest = order[k:]
        for rank, idx in enumerate(rest):
            if len(rest) > 1:
                out[idx] = negative_hi - (
                    rank / max(len(rest) - 1, 1)
                ) * (negative_hi - negative_lo)
            else:
                out[idx] = negative_lo
        return np.clip(out, 0.0, 1.0)

    def predict_chunk_scores(self, chunks):
        if not chunks:
            return []
        raw = self._blend(self._rows(chunks))
        scores = self._safe_topk(raw, "band" if SAFETY_MODE == "band" else "honest")
        return [round(float(score), 6) for score in scores]

    def predict_chunk_score(self, chunk):
        scores = self.predict_chunk_scores([chunk])
        return scores[0] if scores else 0.5
