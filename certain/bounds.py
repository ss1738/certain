"""How much the coverage guarantee moves with the calibration draw.

Split conformal's guarantee is *marginal*: coverage is at least ``1 - alpha``
averaged over calibration sets. Any one calibration set gives a predictor whose
actual coverage is a random variable, and on a small set that variable is wide.
A table reporting "coverage 0.93" from 40 examples is not telling you what it
looks like it is telling you.

For continuous scores the coverage conditional on the calibration set is exactly
Beta distributed:

    coverage | calibration  ~  Beta(k, n + 1 - k),   k = ceil((n + 1)(1 - alpha))

so a lower confidence bound on it is a Beta quantile. This is the number to put
in front of anyone who has to sign off on the system, because it answers "how bad
could this particular fitted model be?" rather than "how good is the method on
average?".

The regularised incomplete beta function is implemented here with a continued
fraction, and its quantile by bisection, so the module stays dependency-free.

Reference: Vovk, "Conditional validity of inductive conformal predictors" (ACML
2012), on training-conditional validity.
"""

from __future__ import annotations

import math

_FPMIN = 1e-300


def _betacf(a: float, b: float, x: float, itmax: int = 300) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``, the Beta(a, b) CDF."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be > 0")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse Beta(a, b) CDF, by bisection on the monotone CDF."""
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def training_conditional_bound(
    n: int, alpha: float = 0.1, confidence: float = 0.95
) -> float:
    """Lower bound on this calibration set's coverage, at the given confidence.

    With probability ``confidence`` over the draw of ``n`` calibration points, a
    split-conformal predictor fitted on them covers at least the returned rate.
    Small ``n`` gives a bound far below ``1 - alpha``: that gap is the honest
    uncertainty, not a defect.

    Returns 0.0 when ``n`` is too small for the quantile to be finite, which is
    the same floor as the conformal threshold itself (``n >= ceil(1/alpha) - 1``).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return 0.0  # threshold is vacuous; nothing to promise
    # coverage ~ Beta(k, n + 1 - k); we want its lower (1 - confidence) quantile
    return beta_quantile(1.0 - confidence, float(k), float(n + 1 - k))
