"""Tests for conformal sets and selective answering."""

import random

import pytest

from certain import ConformalClassifier, SelectiveAnswerer

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
    # Confidence correlates with correctness.
    confidences, correct = [], []
    for _ in range(3000):
        conf = rng.random()
        correct.append(rng.random() < 0.4 + 0.55 * conf)  # more confident -> more right
        confidences.append(conf)
    ans = SelectiveAnswerer(target_error=0.1)
    ans.calibrate(confidences, correct)
    stats = ans.evaluate(confidences, correct)
    assert stats["error_on_answered"] <= 0.15  # near target, with finite-sample slack
    assert 0.0 < stats["coverage"] <= 1.0


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
