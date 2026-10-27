"""Adaptive Conformal Inference: long-run coverage under distribution shift.

Split conformal assumes exchangeability, which production traffic violates the
moment prompts, retrieval quality or user mix drift. When that happens a
threshold fitted once silently stops covering at the rate it promises, and
nothing in the output says so.

Adaptive Conformal Inference fixes it with one line of feedback. After each item
whose outcome you learn, nudge the working miscoverage level:

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

where ``err_t`` is 1 when the true label fell outside the set and 0 otherwise.
Miss and ``alpha_t`` drops, so the next set is wider; cover and it drifts back up,
so sets stay as small as the data allows. Gibbs & Candes prove

    | (1/T) * sum_t err_t  -  alpha |  <=  (alpha_1 + gamma) / (T * gamma)

for *any* sequence, including adversarial ones, with no exchangeability
assumption. Long-run empirical coverage converges to the target whatever the
drift does. ``alpha_t`` is deliberately not clamped to [0, 1]: it is
self-correcting, because above 1 the set is empty, which scores an error and
pushes it straight back down, and that behaviour is what the bound relies on.

Scope: this adapts the working alpha, not the nonconformity distribution, which
stays as calibrated. That is the cheap and honest version. It corrects the rate
at which you miss, and it cannot repair a scoring function that has gone stale
in a way the calibration set never saw. ``gamma`` trades responsiveness against
stability: larger reacts faster to a real shift and jitters more on noise.

Reference: Gibbs & Candes, "Adaptive Conformal Inference Under Distribution
Shift" (arXiv:2106.00170), NeurIPS 2021.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence

from .classifier import _normalise


class AdaptiveConformalClassifier:
    """Conformal sets whose working alpha adapts to observed misses.

    Args:
        alpha: Target long-run miscoverage rate.
        gamma: Step size. Larger adapts faster and is noisier.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.01) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not (gamma > 0.0) or math.isnan(gamma) or math.isinf(gamma):
            raise ValueError("gamma must be finite and > 0")
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha
        self.steps = 0
        self.errors = 0
        self._nonconf: list[float] | None = None

    def calibrate(
        self,
        score_rows: Sequence[Mapping[Hashable, float]],
        labels: Sequence[Hashable],
    ) -> None:
        """Fit the nonconformity distribution from held-out (scores, label) pairs."""
        if len(score_rows) != len(labels):
            raise ValueError("score_rows and labels must be the same length")
        if not score_rows:
            raise ValueError("need at least one calibration example")
        self._nonconf = sorted(
            1.0 - _normalise(row).get(y, 0.0) for row, y in zip(score_rows, labels)
        )

    def _qhat(self, alpha: float) -> float:
        nonconf = self._nonconf
        if nonconf is None:
            raise RuntimeError("call calibrate() first")
        n = len(nonconf)
        k = math.ceil((n + 1) * (1.0 - alpha))
        if k > n:
            return 1.0   # include every label
        if k < 1:
            return -1.0  # include none; nonconformity is never negative
        return nonconf[k - 1]

    def predict_set(self, scores: Mapping[Hashable, float]) -> set:
        """The label set at the *current* working alpha."""
        qhat = self._qhat(self.alpha_t)
        probs = _normalise(scores)
        return {label for label, p in probs.items() if (1.0 - p) <= qhat}

    def record(self, covered: bool) -> float:
        """Feed back whether the last set covered the truth; returns the new alpha."""
        self.steps += 1
        err = 0.0 if covered else 1.0
        self.errors += int(not covered)
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
        return self.alpha_t

    def observe(
        self, scores: Mapping[Hashable, float], label: Hashable
    ) -> set:
        """Predict a set and immediately record whether it covered ``label``."""
        s = self.predict_set(scores)
        self.record(label in s)
        return s

    @property
    def error_rate(self) -> float:
        """Empirical miscoverage so far. Converges to ``alpha``."""
        return self.errors / self.steps if self.steps else 0.0

    def evaluate(
        self,
        score_rows: Sequence[Mapping[Hashable, float]],
        labels: Sequence[Hashable],
    ) -> dict[str, float]:
        """Stream a split through, adapting as it goes."""
        start_steps, start_errors = self.steps, self.errors
        total_size = 0
        for row, y in zip(score_rows, labels):
            total_size += len(self.observe(row, y))
        seen = self.steps - start_steps or 1
        missed = self.errors - start_errors
        return {
            "coverage": 1.0 - missed / seen,
            "avg_set_size": total_size / seen,
            "target_coverage": 1.0 - self.alpha,
            "final_alpha_t": self.alpha_t,
        }
