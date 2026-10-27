"""Regenerate docs/benchmark.png from the benchmark in benchmarks/bench_coverage.py.

Kept as a script so the chart in the README is reproducible rather than drawn by
hand. matplotlib is a docs-time dependency only; the library itself stays
dependency-free.

Run: python docs/make_benchmark_chart.py
"""

from __future__ import annotations

import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

from bench_coverage import (
    ALPHAS,
    REPEATS,
    N,
    mean,
    synth,
    top1_coverage,
)

from certain import ConformalClassifier


def collect():
    naive, results = [], {a: [] for a in ALPHAS}
    for seed in range(REPEATS):
        rng = random.Random(seed)
        cal_rows, cal_labels = synth(N, rng)
        test_rows, test_labels = synth(N, rng)
        naive.append(top1_coverage(test_rows, test_labels))
        for alpha in ALPHAS:
            clf = ConformalClassifier(alpha=alpha)
            clf.calibrate(cal_rows, cal_labels)
            results[alpha].append(clf.evaluate(test_rows, test_labels)["coverage"])
    return mean(naive), results


def main() -> None:
    naive, results = collect()
    targets = [100 * (1 - a) for a in ALPHAS]
    means = [100 * mean(results[a]) for a in ALPHAS]
    lo = [m - 100 * min(results[a]) for m, a in zip(means, ALPHAS)]
    hi = [100 * max(results[a]) - m for m, a in zip(means, ALPHAS)]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=120)

    ax.plot([76, 101], [76, 101], ls="--", color="#b0b0b0", lw=1.4, zorder=1)
    ax.annotate("perfect calibration", xy=(84, 84), rotation=37,
                color="#8a8a8a", fontsize=9.5, ha="center", va="bottom")

    ax.axhline(100 * naive, ls="--", color="#e8743b", lw=1.6, zorder=2)
    ax.annotate(f"naive top-1: {100 * naive:.1f}% (no guarantee)",
                xy=(90.5, 100 * naive + 0.5), color="#c9601f", fontsize=9.5)

    ax.errorbar(targets, means, yerr=[lo, hi], fmt="o-", color="#3b76af",
                ecolor="#3b76af", elinewidth=1.4, capsize=4, markersize=7,
                lw=2.2, zorder=3)

    for t, m in zip(targets, means):
        ax.annotate(f"{m:.1f}", xy=(t, m), xytext=(5, -13),
                    textcoords="offset points", fontsize=10, fontweight="bold")

    ax.set_xlabel("target coverage (%)", fontsize=11)
    ax.set_ylabel("empirical coverage (%)", fontsize=11)
    ax.set_title("certain: conformal coverage tracks the target",
                 fontsize=14, fontweight="bold", loc="left", pad=22)
    ax.annotate(f"mean of {REPEATS} independent calibration splits; "
                "bars show the per-run range",
                xy=(0, 1.015), xycoords="axes fraction", fontsize=10,
                color="#5a5a5a")
    ax.set_xlim(76.5, 101)
    ax.set_ylim(76.5, 101)

    out = os.path.join(os.path.dirname(__file__), "benchmark.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
