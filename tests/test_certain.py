"""Tests for conformal sets and selective answering."""

import random

import pytest

from certain import (
    AdaptiveConformalClassifier,
    ConformalClassifier,
    ConformalRiskController,
    MondrianConformalClassifier,
    SelectiveAnswerer,
    training_conditional_bound,
)

CLASSES = ["a", "b", "c", "d", "e"]


def _synth(n, rng, sharpness=2.0):
    """Noisy multiclass scores: the true class tends to score higher."""
    rows, labels = [], []
    for _ in range(n):
        y = rng.choice(CLASSES)
        row = {c: rng.random() for c in CLASSES}
        row[y] += sharpness * rng.random()  # signal, but noisy/overlapping
        rows.append(row)
        labels.append(y)
    return rows, labels


def test_alpha_validation():
    with pytest.raises(ValueError):
        ConformalClassifier(alpha=0.0)
    with pytest.raises(ValueError):
        ConformalClassifier(alpha=1.0)


def test_calibrate_length_mismatch():
    with pytest.raises(ValueError):
        ConformalClassifier().calibrate([{"a": 1.0}], ["a", "b"])


def test_predict_before_calibrate_raises():
    with pytest.raises(RuntimeError):
        ConformalClassifier().predict_set({"a": 1.0})


def test_conformal_coverage_guarantee():
    rng = random.Random(0)
    cal_rows, cal_labels = _synth(2000, rng)
    test_rows, test_labels = _synth(2000, rng)
    for alpha in (0.05, 0.1, 0.2):
        clf = ConformalClassifier(alpha=alpha)
        clf.calibrate(cal_rows, cal_labels)
        stats = clf.evaluate(test_rows, test_labels)
        # Marginal coverage should be at least 1 - alpha (small slack for noise).
        assert stats["coverage"] >= (1.0 - alpha) - 0.04
        # Sets may be empty at loose alpha (valid abstention), so only require > 0.
        assert stats["avg_set_size"] > 0.0


def test_smaller_alpha_gives_larger_sets():
    rng = random.Random(1)
    rows, labels = _synth(1500, rng)
    loose = ConformalClassifier(alpha=0.3)
    tight = ConformalClassifier(alpha=0.02)
    loose.calibrate(rows, labels)
    tight.calibrate(rows, labels)
    s_loose = loose.evaluate(rows, labels)["avg_set_size"]
    s_tight = tight.evaluate(rows, labels)["avg_set_size"]
    assert s_tight >= s_loose


def test_predict_returns_top_label():
    clf = ConformalClassifier()
    assert clf.predict({"a": 0.1, "b": 0.9, "c": 0.0}) == "b"


def test_selective_controls_error_on_answered():
    rng = random.Random(2)

    def synth(n):
        confs, ok = [], []
        for _ in range(n):
            conf = rng.random()
            ok.append(rng.random() < 0.4 + 0.55 * conf)  # more confident -> more right
            confs.append(conf)
        return confs, ok

    cal_c, cal_ok = synth(3000)
    test_c, test_ok = synth(3000)
    ans = SelectiveAnswerer(target_error=0.1)
    ans.calibrate(cal_c, cal_ok)
    stats = ans.evaluate(test_c, test_ok)  # held-out split, not the calibration data
    assert stats["error_on_answered"] <= 0.15  # near target, with finite-sample slack
    assert 0.0 < stats["coverage"] <= 1.0


def test_selective_is_tie_safe():
    # Tied confidences must not let answered-error exceed the target in-sample.
    # The naive item-by-item sweep would pick threshold 0.5 and answer all 10
    # (error 0.5); the level-based sweep abstains instead.
    ans = SelectiveAnswerer(target_error=0.1)
    conf = [0.5] * 10
    correct = [True] * 5 + [False] * 5
    ans.calibrate(conf, correct)
    stats = ans.evaluate(conf, correct)
    assert stats["error_on_answered"] <= 0.1


def test_selective_abstains_when_impossible():
    ans = SelectiveAnswerer(target_error=0.0)
    # Every item wrong -> no threshold satisfies 0% error -> abstain on everything.
    ans.calibrate([0.9, 0.8, 0.7], [False, False, False])
    assert ans.answer(1.0) is False


def test_selective_validation():
    with pytest.raises(ValueError):
        SelectiveAnswerer(target_error=1.5)
    with pytest.raises(RuntimeError):
        SelectiveAnswerer().answer(0.5)


def test_small_alpha_needs_enough_calibration():
    """A target of 1-alpha needs >= ceil(1/alpha)-1 calibration examples.

    Pins the documented caveat in README "Calibration size and small alpha".
    Below that size the conformal quantile saturates at 1.0 and every set holds
    every label, so coverage is trivially 1.0 and the set carries no information.
    """
    import math

    rng = random.Random(0)
    for alpha in (0.2, 0.1, 0.05, 0.01):
        n_min = math.ceil(1 / alpha) - 1

        rows, labels = _synth(n_min - 1, rng)
        assert ConformalClassifier(alpha=alpha).calibrate(rows, labels) == 1.0

        rows, labels = _synth(n_min, rng)
        assert ConformalClassifier(alpha=alpha).calibrate(rows, labels) < 1.0


def test_vacuous_sets_below_minimum_calibration_size():
    """Under-calibrated at alpha=0.01: every label is in every set."""
    rng = random.Random(1)
    rows, labels = _synth(50, rng)  # < 99, the minimum for alpha=0.01
    clf = ConformalClassifier(alpha=0.01)
    clf.calibrate(rows, labels)
    assert clf.predict_set({c: rng.random() for c in CLASSES}) == set(CLASSES)


def test_coverage_guarantee_holds_on_average():
    """The guarantee is marginal: it holds in expectation, not per run.

    A single split lands either side of the target, so this averages over
    independent calibration draws, which is what the theory actually bounds.
    """
    alpha = 0.1
    covs = []
    for seed in range(20):
        rng = random.Random(100 + seed)
        clf = ConformalClassifier(alpha=alpha)
        clf.calibrate(*_synth(600, rng))
        covs.append(clf.evaluate(*_synth(600, rng))["coverage"])
    assert sum(covs) / len(covs) >= 1 - alpha


# --- Conformal Risk Control -------------------------------------------------


def _loss_matrix(n_high, n_low, lambdas):
    """Rows whose loss drops to 0 at a different lambda for two groups."""
    high = [[1.0 if lam < 0.5 else 0.0 for lam in lambdas] for _ in range(n_high)]
    low = [[1.0 if lam < 0.25 else 0.0 for lam in lambdas] for _ in range(n_low)]
    return high + low


def test_risk_controller_picks_smallest_controlling_lambda():
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = _loss_matrix(50, 50, lambdas)
    c = ConformalRiskController(alpha=0.1)
    assert c.calibrate(rows, lambdas) == 0.5
    assert c.evaluate(rows, lambdas)["risk"] <= 0.1


def test_risk_controller_recovers_split_conformal_on_indicator_loss():
    """With a miscoverage indicator loss, CRC controls risk at alpha."""
    rng = random.Random(3)
    lambdas = [i / 20 for i in range(21)]
    # Item is "covered" once lambda passes its own difficulty threshold.
    rows = []
    for _ in range(500):
        d = rng.random()
        rows.append([1.0 if lam < d else 0.0 for lam in lambdas])
    c = ConformalRiskController(alpha=0.1)
    c.calibrate(rows, lambdas)
    test = []
    for _ in range(500):
        d = rng.random()
        test.append([1.0 if lam < d else 0.0 for lam in lambdas])
    assert c.evaluate(test, lambdas)["risk"] <= 0.1 + 0.05


def test_risk_controller_rejects_non_monotone_loss():
    """The guarantee needs non-increasing loss, so a violation must not pass."""
    lambdas = [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="non-increasing"):
        ConformalRiskController(alpha=0.1).calibrate([[0.0, 1.0, 0.0]], lambdas)


def test_risk_controller_rejects_bad_input():
    lambdas = [0.0, 0.5, 1.0]
    c = ConformalRiskController(alpha=0.1)
    with pytest.raises(ValueError):
        c.calibrate([[1.0, 0.0, 0.0]], [1.0, 0.5, 0.0])      # descending lambdas
    with pytest.raises(ValueError):
        c.calibrate([[float("nan"), 0.0, 0.0]], lambdas)     # NaN
    with pytest.raises(ValueError):
        c.calibrate([[5.0, 0.0, 0.0]], lambdas)              # above loss_bound
    with pytest.raises(ValueError):
        c.calibrate([[1.0, 1.0, 1.0]], lambdas)              # nothing controls it
    with pytest.raises(RuntimeError):
        ConformalRiskController().evaluate([[1.0, 0.0, 0.0]], lambdas)


# --- Adaptive Conformal Inference -------------------------------------------


def test_adaptive_holds_coverage_under_shift_where_static_fails():
    """The point of ACI: static conformal collapses under drift, adaptive does not."""
    def gen(n, rng, sharpness):
        return _synth(n, rng, sharpness=sharpness)

    rng = random.Random(0)
    cal = gen(1000, rng, 2.0)

    adaptive = AdaptiveConformalClassifier(alpha=0.1, gamma=0.02)
    adaptive.calibrate(*cal)
    adaptive.evaluate(*gen(2000, rng, 2.0))          # settle on in-distribution data
    shifted = adaptive.evaluate(*gen(3000, rng, 0.3))  # model degrades hard

    rng2 = random.Random(0)
    static = ConformalClassifier(alpha=0.1)
    static.calibrate(*gen(1000, rng2, 2.0))
    gen(2000, rng2, 2.0)
    static_shifted = static.evaluate(*gen(3000, rng2, 0.3))

    assert static_shifted["coverage"] < 0.7          # measured ~0.54
    assert shifted["coverage"] > 0.85                # measured ~0.90
    # Adaptation is paid for in set size, not silently.
    assert shifted["avg_set_size"] > static_shifted["avg_set_size"]


def test_adaptive_matches_target_without_shift():
    rng = random.Random(7)
    m = AdaptiveConformalClassifier(alpha=0.1, gamma=0.02)
    m.calibrate(*_synth(800, rng))
    res = m.evaluate(*_synth(3000, rng))
    assert abs(res["coverage"] - 0.9) < 0.05


def test_adaptive_alpha_is_self_correcting_and_unclamped():
    """alpha_t may leave [0, 1]; an empty set scores an error and pulls it back."""
    rng = random.Random(1)
    m = AdaptiveConformalClassifier(alpha=0.1, gamma=0.3)
    m.calibrate(*_synth(100, rng))
    for _ in range(10):
        m.record(covered=False)      # forced misses drive alpha_t down
    assert m.alpha_t < 0.0
    assert m.predict_set({c: 1.0 for c in CLASSES}) == set(CLASSES)  # covers all
    # Recovery is deliberately slow: each covered step adds only gamma * alpha,
    # so climbing back from 10 forced misses takes ~90 of them, not 10.
    before = m.alpha_t
    for _ in range(100):
        m.record(covered=True)
    assert m.alpha_t > before
    assert m.alpha_t > 0.0
    assert m.error_rate == 10 / 110


def test_adaptive_validation():
    with pytest.raises(ValueError):
        AdaptiveConformalClassifier(alpha=0.0)
    with pytest.raises(ValueError):
        AdaptiveConformalClassifier(gamma=0.0)
    with pytest.raises(RuntimeError):
        AdaptiveConformalClassifier().predict_set({"a": 1.0})
    with pytest.raises(ValueError):
        AdaptiveConformalClassifier().calibrate([{"a": 1.0}], ["a", "b"])


# --- Mondrian group-conditional conformal -----------------------------------


def _grouped(n_easy, n_hard, rng):
    """A majority group the scorer handles well and a minority it does not."""
    easy = _synth(n_easy, rng, sharpness=3.0)
    hard = _synth(n_hard, rng, sharpness=0.15)
    return (
        easy[0] + hard[0],
        easy[1] + hard[1],
        ["easy"] * n_easy + ["hard"] * n_hard,
    )


def test_mondrian_covers_each_group_where_marginal_hides_a_failure():
    """The whole point: a marginal number can look fine while a group is failing."""
    rng = random.Random(0)
    cal = _grouped(1800, 200, rng)
    test = _grouped(1800, 200, rng)

    mondrian = MondrianConformalClassifier(alpha=0.1)
    mondrian.calibrate(*cal)
    res = mondrian.evaluate(*test)
    assert res["worst_group_coverage"] > 0.85          # measured ~0.90

    marginal = ConformalClassifier(alpha=0.1)
    marginal.calibrate(cal[0], cal[1])
    hard_hits = [
        1 if y in marginal.predict_set(row) else 0
        for row, y, g in zip(*test) if g == "hard"
    ]
    hard_coverage = sum(hard_hits) / len(hard_hits)
    assert hard_coverage < 0.6                          # measured ~0.47
    # ...while the marginal headline looks healthy.
    assert marginal.evaluate(test[0], test[1])["coverage"] > 0.85


def test_mondrian_flags_vacuous_small_groups():
    rng = random.Random(2)
    rows, labels = _synth(200, rng)
    groups = ["big"] * 195 + ["tiny"] * 5      # tiny < ceil(1/0.1)-1 = 9
    m = MondrianConformalClassifier(alpha=0.1)
    m.calibrate(rows, labels, groups)
    assert m.min_calibration_size == 9
    table = {r["group"]: r for r in m.calibration_table()}
    assert table["tiny"]["vacuous"] is True
    assert table["big"]["vacuous"] is False
    # A tiny group's threshold admits everything, so it promises nothing useful.
    assert m.predict_set({c: random.random() for c in CLASSES}, "tiny") == set(CLASSES)
    # And its honest lower bound says so.
    assert table["tiny"]["coverage_lower_bound"] == 0.0
    assert table["big"]["coverage_lower_bound"] > 0.8


def test_mondrian_refuses_unseen_group():
    rng = random.Random(3)
    rows, labels = _synth(100, rng)
    m = MondrianConformalClassifier(alpha=0.1)
    m.calibrate(rows, labels, ["x"] * 100)
    with pytest.raises(KeyError, match="no calibration data for group"):
        m.predict_set({c: 1.0 for c in CLASSES}, "unseen")
    # evaluate counts them rather than scoring them against someone else.
    res = m.evaluate(rows, labels, ["unseen"] * 100)
    assert res["skipped_unknown_group"] == 100


def test_mondrian_validation():
    with pytest.raises(ValueError):
        MondrianConformalClassifier(alpha=1.0)
    with pytest.raises(ValueError):
        MondrianConformalClassifier().calibrate([{"a": 1.0}], ["a"], ["g", "h"])
    with pytest.raises(RuntimeError):
        MondrianConformalClassifier().predict_set({"a": 1.0}, "g")
    with pytest.raises(RuntimeError):
        MondrianConformalClassifier().calibration_table()


# --- Training-conditional bound ---------------------------------------------


def test_training_conditional_bound_matches_simulation():
    """Coverage | calibration ~ Beta(k, n+1-k); the bound must track reality."""
    alpha = 0.1
    for n, floor in ((100, 0.80), (500, 0.86)):
        covs = []
        for s in range(150):
            rng = random.Random(500 + s)
            c = ConformalClassifier(alpha=alpha)
            c.calibrate(*_synth(n, rng))
            covs.append(c.evaluate(*_synth(1500, rng))["coverage"])
        bound = training_conditional_bound(n, alpha, confidence=0.95)
        assert bound > floor
        assert bound < 1 - alpha            # a bound, not the target itself
        # At most a few of 150 draws should fall below a 95% lower bound.
        assert sum(1 for c_ in covs if c_ < bound) <= 20


def test_training_conditional_bound_tightens_with_n():
    prev = 0.0
    for n in (20, 100, 1000, 10000):
        b = training_conditional_bound(n, 0.1, 0.95)
        assert b > prev
        prev = b
    assert training_conditional_bound(10000, 0.1, 0.95) < 0.9
    assert training_conditional_bound(5, 0.1, 0.95) == 0.0   # below the floor


def test_training_conditional_bound_validation():
    with pytest.raises(ValueError):
        training_conditional_bound(0, 0.1)
    with pytest.raises(ValueError):
        training_conditional_bound(100, 0.0)
    with pytest.raises(ValueError):
        training_conditional_bound(100, 0.1, confidence=1.0)
