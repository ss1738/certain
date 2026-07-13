"""Split-conformal prediction sets with a finite-sample coverage guarantee.

A model gives per-label scores; this turns them into a *set* of labels that
contains the true label with probability at least ``1 - alpha`` -- guaranteed on
exchangeable data, distribution-free, from a held-out calibration set. When the
model is confident the set is a singleton (answer it); when it isn't, the set
grows (a calibrated "I'm not sure between these"), which is exactly the signal you
want before trusting an LLM/classifier decision.

Reference: Vovk et al.; Angelopoulos & Bates, "A Gentle Introduction to Conformal
Prediction" (2021).
"""

from __future__ import annotations

import math
from typing import Any, Hashable, Mapping, Sequence


def _normalise(scores: Mapping[Hashable, float]) -> dict[Hashable, float]:
    """Turn non-negative scores into a distribution (sum 1)."""
    items = {k: max(0.0, float(v)) for k, v in scores.items()}
    total = sum(items.values())
    if total <= 0.0:
        n = len(items) or 1
        return {k: 1.0 / n for k in items}
    return {k: v / total for k, v in items.items()}


class ConformalClassifier:
    """Calibrated prediction sets with >= 1 - alpha coverage.

    Args:
        alpha: Target miscoverage rate (e.g. 0.1 for 90% coverage).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.qhat: float | None = None

    def calibrate(
        self,
        score_rows: Sequence[Mapping[Hashable, float]],
        labels: Sequence[Hashable],
    ) -> float:
        """Fit the conformal threshold from held-out (scores, true label) pairs."""
        if len(score_rows) != len(labels):
            raise ValueError("score_rows and labels must be the same length")
        if not score_rows:
            raise ValueError("need at least one calibration example")
        # Nonconformity of the true label = 1 - its normalised probability.
        nonconf = sorted(
            1.0 - _normalise(row).get(y, 0.0) for row, y in zip(score_rows, labels)
        )
        n = len(nonconf)
        k = math.ceil((n + 1) * (1.0 - self.alpha))
        # k > n means the quantile is +inf -> include every label (coverage 1).
        self.qhat = nonconf[k - 1] if k <= n else 1.0
        return self.qhat

    def predict_set(self, scores: Mapping[Hashable, float]) -> set:
        """Return the calibrated label set for one query."""
        if self.qhat is None:
            raise RuntimeError("call calibrate() first")
        probs = _normalise(scores)
        return {label for label, p in probs.items() if (1.0 - p) <= self.qhat}

    def predict(self, scores: Mapping[Hashable, float]) -> Any:
        """The single most likely label (ignores the set / guarantee).

        Returns ``None`` for an empty score mapping.
        """
        if not scores:
            return None
        return max(_normalise(scores).items(), key=lambda kv: kv[1])[0]

    def evaluate(
        self,
        score_rows: Sequence[Mapping[Hashable, float]],
        labels: Sequence[Hashable],
    ) -> dict[str, float]:
        """Empirical coverage and average set size on a test split."""
        covered = 0
        total_size = 0
        for row, y in zip(score_rows, labels):
            s = self.predict_set(row)
            covered += y in s
            total_size += len(s)
        n = len(labels) or 1
        return {
            "coverage": covered / n,
            "avg_set_size": total_size / n,
            "target_coverage": 1.0 - self.alpha,
        }
