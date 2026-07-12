"""Selective prediction -- answer when confident, abstain when not.

Given a calibration set of (confidence, was_correct) pairs, pick the lowest
confidence threshold at which the error rate among answered items stays within a
target. At serve time, answer only above that threshold. This is the practical
"should the LLM answer or say I don't know" gate.

Note on guarantees: the threshold controls the *empirical* error on calibration
and holds approximately on exchangeable test data. For a strict distribution-free
bound, use :class:`~certain.classifier.ConformalClassifier` (which has a
finite-sample coverage guarantee) or add a Learn-then-Test correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class SelectiveAnswerer:
    """Confidence-thresholded answer/abstain gate.

    Args:
        target_error: Maximum tolerated error rate among answered items.
    """

    target_error: float = 0.05
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.target_error <= 1.0):
            raise ValueError("target_error must be in [0, 1]")

    def calibrate(
        self, confidences: Sequence[float], correct: Sequence[bool]
    ) -> float:
        """Fit the lowest threshold keeping calibration error <= target_error.

        Lower threshold = more items answered, so we take the most permissive
        threshold that still satisfies the target -- maximising coverage.
        """
        if len(confidences) != len(correct):
            raise ValueError("confidences and correct must be the same length")
        # Sort most-confident first; sweep, tracking cumulative error.
        order = sorted(zip(confidences, correct), key=lambda x: -x[0])
        wrong = 0
        chosen: float | None = None
        for i, (conf, ok) in enumerate(order, start=1):
            if not ok:
                wrong += 1
            if wrong / i <= self.target_error:
                chosen = conf  # can safely answer down to this confidence
        # None found -> threshold above any confidence (abstain on everything).
        self.threshold = chosen if chosen is not None else float("inf")
        return self.threshold

    def answer(self, confidence: float) -> bool:
        """True = answer, False = abstain."""
        if self.threshold is None:
            raise RuntimeError("call calibrate() first")
        return confidence >= self.threshold

    def evaluate(
        self, confidences: Sequence[float], correct: Sequence[bool]
    ) -> dict[str, float]:
        """Coverage (fraction answered) and error among answered, on a test split."""
        answered = [ok for conf, ok in zip(confidences, correct) if self.answer(conf)]
        n = len(confidences) or 1
        errors = sum(1 for ok in answered if not ok)
        return {
            "coverage": len(answered) / n,
            "error_on_answered": errors / len(answered) if answered else 0.0,
            "target_error": self.target_error,
        }
