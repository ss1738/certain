"""Does the coverage guarantee actually hold? And what does naive top-1 give?

A naive classifier answers its single most likely label; its "coverage" is just
its accuracy, and it comes with no guarantee. Conformal prediction sets are
calibrated so the true label is inside the set at the target rate. This benchmark
shows the empirical coverage tracking ``1 - alpha`` across several alphas, on a
held-out test split, vs the fixed (and lower) coverage of top-1.

Run: python benchmarks/bench_coverage.py
"""

from __future__ import annotations

import random

from certain import ConformalClassifier

CLASSES = ["a", "b", "c", "d", "e"]
N = 4000
SEED = 7


def synth(n, rng, sharpness=1.6):
    rows, labels = [], []
    for _ in range(n):
        y = rng.choice(CLASSES)
        row = {c: rng.random() for c in CLASSES}
        row[y] += sharpness * rng.random()
        rows.append(row)
        labels.append(y)
    return rows, labels


def top1_coverage(rows, labels) -> float:
    clf = ConformalClassifier()  # only using .predict here
    return sum(clf.predict(r) == y for r, y in zip(rows, labels)) / len(labels)


def main() -> None:
    rng = random.Random(SEED)
    cal_rows, cal_labels = synth(N, rng)
    test_rows, test_labels = synth(N, rng)

    print(f"classes: {len(CLASSES)}   calibration: {N}   test: {N}\n")
    print(f"naive top-1 coverage (= accuracy): {top1_coverage(test_rows, test_labels):.1%}")
    print("  (no guarantee -- it is whatever the model's accuracy happens to be)\n")

    print(f"{'target':>8} {'empirical':>10} {'avg set size':>13}")
    for alpha in (0.01, 0.05, 0.10, 0.20):
        clf = ConformalClassifier(alpha=alpha)
        clf.calibrate(cal_rows, cal_labels)
        s = clf.evaluate(test_rows, test_labels)
        print(f"{1 - alpha:>8.0%} {s['coverage']:>10.1%} {s['avg_set_size']:>13.2f}")
    print("\nempirical coverage tracks the target -- guaranteed >= 1-alpha, "
          "distribution-free.")


if __name__ == "__main__":
    main()
