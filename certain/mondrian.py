"""Mondrian conformal prediction: coverage that holds *within* each group.

A marginal guarantee is an average over everyone. It is satisfied by a model that
covers 97% of your largest group and 60% of a small one, and the single headline
number will not show you that. Wherever the groups mean something, a tenant, a
document source, a language, a clinical site, a protected attribute, the average
is the wrong object to promise.

Mondrian conformal prediction fits a separate threshold per group, so

    P(Y in C(X) | group(X) = g)  >=  1 - alpha     for every group g

each with the same finite-sample, distribution-free argument as the marginal
version. The cost is that every group has to carry its own calibration set, and a
group too small to support the quantile gets a vacuous threshold rather than a
quiet approximation. :meth:`MondrianConformalClassifier.calibration_table` names
those groups instead of hiding them, and pairs each with the training-conditional
lower bound from :mod:`certain.bounds`, which is what a small group's number is
actually worth.

Note the guarantee is conditional on the groups you declare, and on nothing else.
Picking the grouping is a modelling decision this library cannot make for you: a
grouping that misses the axis along which the model actually degrades will look
reassuring and mean little.

Reference: Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*,
on Mondrian and label-conditional conformal predictors.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence

from .bounds import training_conditional_bound
from .classifier import _normalise


class MondrianConformalClassifier:
    """Split conformal fitted separately within each group.

    Args:
        alpha: Target miscoverage rate, applied within every group.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.qhat: dict[Hashable, float] = {}
        self._counts: dict[Hashable, int] = {}

    @property
    def min_calibration_size(self) -> int:
        """Smallest per-group calibration set with a non-vacuous threshold."""
        return math.ceil(1.0 / self.alpha) - 1

    def calibrate(
        self,
        score_rows: Sequence[Mapping[Hashable, float]],
        labels: Sequence[Hashable],
        groups: Sequence[Hashable],
    ) -> dict[Hashable, float]:
        """Fit one conformal threshold per group.

        Returns:
            The fitted threshold for each group. A threshold of 1.0 means that
            group had too little calibration data and its sets will contain every
            label; :meth:`calibration_table` flags these as vacuous.
        """
        if not (len(score_rows) == len(labels) == len(groups)):
            raise ValueError("score_rows, labels and groups must be the same length")
        if not score_rows:
            raise ValueError("need at least one calibration example")

        by_group: dict[Hashable, list[float]] = {}
        for row, y, g in zip(score_rows, labels, groups):
            by_group.setdefault(g, []).append(1.0 - _normalise(row).get(y, 0.0))

        self.qhat, self._counts = {}, {}
        for g, nonconf in by_group.items():
            nonconf.sort()
            n = len(nonconf)
            k = math.ceil((n + 1) * (1.0 - self.alpha))
            self.qhat[g] = nonconf[k - 1] if k <= n else 1.0
            self._counts[g] = n
        return dict(self.qhat)

    def predict_set(self, scores: Mapping[Hashable, float], group: Hashable) -> set:
        """The calibrated label set for one query, using its group's threshold."""
        if not self.qhat:
            raise RuntimeError("call calibrate() first")
        if group not in self.qhat:
            raise KeyError(
                f"no calibration data for group {group!r}; the per-group "
                "guarantee does not extend to unseen groups. Calibrate on it, "
                "or route it to a marginal ConformalClassifier deliberately."
            )
        q = self.qhat[group]
        probs = _normalise(scores)
        return {label for label, p in probs.items() if (1.0 - p) <= q}

    def calibration_table(self, confidence: float = 0.95) -> list[dict]:
        """Per-group audit table: size, threshold, and what it is actually worth.

        ``coverage_lower_bound`` is the training-conditional bound: with
        probability ``confidence`` over the calibration draw, this group's fitted
        predictor covers at least that rate. For a small group it sits well below
        ``1 - alpha``, which is the honest reading of a small group.
        """
        if not self.qhat:
            raise RuntimeError("call calibrate() first")
        rows = []
        for g in sorted(self.qhat, key=repr):
            n = self._counts[g]
            vacuous = self.qhat[g] >= 1.0 and n < self.min_calibration_size
            rows.append({
                "group": g,
                "n_calibration": n,
                "qhat": self.qhat[g],
                "target_coverage": 1.0 - self.alpha,
                "coverage_lower_bound": training_conditional_bound(
                    n, self.alpha, confidence
                ),
                "confidence": confidence,
                "vacuous": vacuous,
            })
        return rows

    def evaluate(
        self,
        score_rows: Sequence[Mapping[Hashable, float]],
        labels: Sequence[Hashable],
        groups: Sequence[Hashable],
    ) -> dict:
        """Overall and per-group empirical coverage on a test split.

        Groups with no calibration data are counted in ``skipped`` rather than
        silently scored against someone else's threshold.
        """
        if not (len(score_rows) == len(labels) == len(groups)):
            raise ValueError("score_rows, labels and groups must be the same length")
        per: dict[Hashable, list[int]] = {}
        sizes: dict[Hashable, int] = {}
        skipped = 0
        for row, y, g in zip(score_rows, labels, groups):
            if g not in self.qhat:
                skipped += 1
                continue
            s = self.predict_set(row, g)
            per.setdefault(g, []).append(1 if y in s else 0)
            sizes[g] = sizes.get(g, 0) + len(s)
        scored = sum(len(v) for v in per.values())
        by_group = [
            {
                "group": g,
                "n_test": len(hits),
                "coverage": sum(hits) / len(hits),
                "avg_set_size": sizes[g] / len(hits),
            }
            for g, hits in sorted(per.items(), key=lambda kv: repr(kv[0]))
        ]
        return {
            "coverage": sum(sum(v) for v in per.values()) / scored if scored else 0.0,
            "target_coverage": 1.0 - self.alpha,
            "worst_group_coverage": min((r["coverage"] for r in by_group), default=0.0),
            "skipped_unknown_group": skipped,
            "by_group": by_group,
        }
