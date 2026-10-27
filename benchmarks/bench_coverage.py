"""Does the coverage guarantee actually hold? And what does naive top-1 give?

A naive classifier answers its single most likely label; its "coverage" is just
its accuracy, and it comes with no guarantee. Conformal prediction sets are
calibrated so the true label is inside the set at the target rate.

The guarantee is *marginal*: coverage is at least ``1 - alpha`` in expectation
over the draw of the calibration set, not on every individual run. A single
calibration split lands above or below the target by chance, so averaging one
run tells you very little. This benchmark therefore repeats the whole procedure
over many independent splits and reports the mean, which is the quantity the
theory actually bounds, alongside the per-run spread.

Run: python benchmarks/bench_coverage.py
"""

from __future__ import annotations

import random

from certain import ConformalClassifier

CLASSES = ["a", "b", "c", "d", "e"]
N = 4000
REPEATS = 50
ALPHAS = (0.01, 0.05, 0.10, 0.20)


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


def mean(xs) -> float:
    return sum(xs) / len(xs)


def main() -> None:
    print(f"classes: {len(CLASSES)}   calibration: {N}   test: {N}   "
          f"independent splits: {REPEATS}\n")

    naive = []
    results = {a: {"cov": [], "size": []} for a in ALPHAS}

    for seed in range(REPEATS):
        rng = random.Random(seed)
        cal_rows, cal_labels = synth(N, rng)
        test_rows, test_labels = synth(N, rng)
        naive.append(top1_coverage(test_rows, test_labels))
        for alpha in ALPHAS:
            clf = ConformalClassifier(alpha=alpha)
            clf.calibrate(cal_rows, cal_labels)
            s = clf.evaluate(test_rows, test_labels)
            results[alpha]["cov"].append(s["coverage"])
            results[alpha]["size"].append(s["avg_set_size"])

    print(f"naive top-1 coverage (= accuracy): {mean(naive):.1%}")
    print("  (no guarantee -- it is whatever the model's accuracy happens to be)\n")

    print(f"{'target':>8} {'mean cov':>9} {'per-run range':>18} {'avg set size':>13}")
    for alpha in ALPHAS:
        cov = results[alpha]["cov"]
        lo, hi = min(cov), max(cov)
        print(f"{1 - alpha:>8.0%} {mean(cov):>9.1%} "
              f"{lo:>9.1%} to {hi:<6.1%} {mean(results[alpha]['size']):>13.2f}")

    print("\nMean coverage lands on the target: that is the marginal guarantee,")
    print("distribution-free and finite-sample, assuming exchangeable data.")
    print("Individual runs fall either side of it, which is expected and is why")
    print("a single split is not evidence for or against the guarantee.")


if __name__ == "__main__":
    main()
