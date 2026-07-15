"""Reusable model wrappers for Poker44 miner artifacts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class ProbabilityAveragingEnsemble:
    """Average bot probabilities from already-fit classifiers."""

    def __init__(self, estimators: Sequence[Any], weights: Sequence[float] | None = None):
        if not estimators:
            raise ValueError("ProbabilityAveragingEnsemble requires at least one estimator")
        self.estimators = list(estimators)
        self.weights = list(weights) if weights is not None else None

    def _positive_scores(self, estimator: Any, x: Any) -> np.ndarray:
        if hasattr(estimator, "predict_proba"):
            return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
        scores = np.asarray(estimator.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    def predict_proba(self, x: Any) -> np.ndarray:
        scores = np.vstack([self._positive_scores(estimator, x) for estimator in self.estimators])
        if self.weights is None:
            positive = scores.mean(axis=0)
        else:
            weights = np.asarray(self.weights, dtype=float)
            weights = weights / weights.sum()
            positive = np.average(scores, axis=0, weights=weights)
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


class CalibratedProbabilityModel:
    """Apply a monotonic safety calibration to a fitted probability model."""

    def __init__(
        self,
        base_model: Any,
        *,
        scale: float = 1.0,
        power: float = 1.0,
        offset: float = 0.0,
        cap: float | None = None,
    ):
        self.base_model = base_model
        self.scale = float(scale)
        self.power = float(power)
        self.offset = float(offset)
        self.cap = float(cap) if cap is not None else None

    def _positive_scores(self, x: Any) -> np.ndarray:
        if hasattr(self.base_model, "predict_proba"):
            return np.asarray(self.base_model.predict_proba(x)[:, 1], dtype=float)
        scores = np.asarray(self.base_model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    def _calibrate(self, scores: np.ndarray) -> np.ndarray:
        positive = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        if self.power != 1.0:
            positive = positive ** self.power
        positive = positive * self.scale + self.offset
        if self.cap is not None:
            positive = np.minimum(positive, self.cap)
        return np.clip(positive, 0.0, 1.0)

    def predict_proba(self, x: Any) -> np.ndarray:
        positive = self._calibrate(self._positive_scores(x))
        return np.column_stack([1.0 - positive, positive])
