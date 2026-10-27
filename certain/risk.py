"""Conformal Risk Control: bound any monotone loss, not just miscoverage.

Split conformal controls one specific loss, the indicator that the true label
fell outside the prediction set. Conformal Risk Control generalises that to any
loss which shrinks as a parameter grows. Given per-item losses ``L_i(lam)``,
non-increasing in ``lam`` and bounded above by ``B``, it fits

    lam_hat = inf{ lam : (n / (n + 1)) * R_n(lam) + B / (n + 1) <= alpha }

where ``R_n(lam)`` is the mean loss over the calibration set. The guarantee is
then ``E[L_{n+1}(lam_hat)] <= alpha`` on a fresh exchangeable item: finite
sample, distribution free, no assumption about the model that produced the
scores. Taking the loss to be the miscoverage indicator recovers split conformal
exactly.

This is the tool for the "how much do I keep?" questions that are not
classification: how many retrieved passages to pass downstream, how many
sub-claims of a generated answer to retain, where to set an abstention cut. It
differs from :class:`~certain.selective.SelectiveAnswerer`, which fixes a
threshold from empirical calibration error and carries no finite-sample bound on
fresh data.

The monotonicity requirement is real, not decorative: the proof needs each item's
loss to be non-increasing in ``lam``, so :meth:`ConformalRiskController.calibrate`
checks it and refuses rather than returning an unsound parameter.

Reference: Angelopoulos, Bates, Fisch, Lei & Schuster, "Conformal Risk Control"
(arXiv:2208.02814).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class ConformalRiskController:
    """Fit the smallest lambda whose expected loss on fresh data is <= alpha.

    Args:
        alpha: Target risk. The fitted lambda satisfies ``E[loss] <= alpha``.
        loss_bound: Upper bound ``B`` on any single item's loss. The default 1.0
            suits indicator and rate losses. Set it to the true maximum: a bound
            that is too small silently breaks the guarantee, so losses above it
            are rejected.
    """

    def __init__(self, alpha: float = 0.1, loss_bound: float = 1.0) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not (loss_bound > 0.0) or math.isinf(loss_bound):
            raise ValueError("loss_bound must be finite and > 0")
        self.alpha = alpha
        self.loss_bound = loss_bound
        self.lambda_: float | None = None
        self._index: int | None = None

    def _check(
        self, loss_rows: Sequence[Sequence[float]], lambdas: Sequence[float]
    ) -> None:
        if len(lambdas) < 1:
            raise ValueError("need at least one candidate lambda")
        if any(
            not (lambdas[j] < lambdas[j + 1]) for j in range(len(lambdas) - 1)
        ):
            raise ValueError("lambdas must be strictly ascending")
        if not loss_rows:
            raise ValueError("need at least one calibration example")
        for i, row in enumerate(loss_rows):
            if len(row) != len(lambdas):
                raise ValueError(f"loss row {i} has {len(row)} entries, "
                                 f"expected {len(lambdas)}")
            for j, v in enumerate(row):
                if math.isnan(v):
                    raise ValueError(f"loss row {i} contains NaN")
                if v < 0.0 or v > self.loss_bound:
                    raise ValueError(
                        f"loss row {i} entry {j} is {v}, outside [0, "
                        f"{self.loss_bound}]"
                    )
                # The guarantee requires each item's loss to be non-increasing
                # in lambda. Accepting a violating row would return a number
                # that looks calibrated and is not.
                if j and v > row[j - 1]:
                    raise ValueError(
                        f"loss row {i} increases from lambda index {j - 1} to "
                        f"{j} ({row[j - 1]} -> {v}); losses must be "
                        "non-increasing in lambda"
                    )

    def calibrate(
        self, loss_rows: Sequence[Sequence[float]], lambdas: Sequence[float]
    ) -> float:
        """Fit lambda from a calibration loss matrix.

        Args:
            loss_rows: One row per calibration item, one column per candidate
                lambda, each row non-increasing along the row.
            lambdas: Candidate values, strictly ascending.

        Returns:
            The fitted lambda.

        Raises:
            ValueError: If no candidate controls the risk. Extend the grid
                upwards: the guarantee needs some lambda whose loss is already
                at or below alpha.
        """
        self._check(loss_rows, lambdas)
        n = len(loss_rows)
        for j, lam in enumerate(lambdas):
            risk = sum(row[j] for row in loss_rows) / n
            if (n / (n + 1)) * risk + self.loss_bound / (n + 1) <= self.alpha:
                self.lambda_ = lam
                self._index = j
                return lam
        raise ValueError(
            "no candidate lambda controls the risk at alpha="
            f"{self.alpha}; extend the lambda grid upwards or raise alpha"
        )

    def evaluate(
        self, loss_rows: Sequence[Sequence[float]], lambdas: Sequence[float]
    ) -> dict[str, float]:
        """Empirical risk at the fitted lambda on a test split."""
        if self._index is None:
            raise RuntimeError("call calibrate() first")
        self._check(loss_rows, lambdas)
        if self._index >= len(lambdas):
            raise ValueError("lambda grid is shorter than the one calibrated on")
        j = self._index
        return {
            "risk": sum(row[j] for row in loss_rows) / len(loss_rows),
            "target_risk": self.alpha,
            "lambda": float(lambdas[j]),
        }
