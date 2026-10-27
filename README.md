# certain

[![CI](https://github.com/ss1738/certain/actions/workflows/ci.yml/badge.svg)](https://github.com/ss1738/certain/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Calibrated confidence and abstention for model outputs.** A softmax number
isn't a probability you can trust. certain turns any score into a **prediction set
with a coverage guarantee** (the true label is inside with probability >= 1-alpha,
distribution-free), a set that **keeps covering when the data drifts**, a
**bound on any monotone loss** rather than just miscoverage, or an
**answer/abstain gate** with a controlled error rate. Model-agnostic, works over
LLM logprobs, judge scores, classifier probabilities, retrieval scores. Pure
Python, zero dependencies.

```python
from certain import ConformalClassifier

clf = ConformalClassifier(alpha=0.1)             # want 90% coverage
clf.calibrate(cal_scores, cal_labels)            # a held-out calibration split

clf.predict_set({"cat": 0.9, "dog": 0.05, "fox": 0.05})   # a singleton set -> confident, answer
clf.predict_set({"cat": 0.4, "dog": 0.35, "fox": 0.25})   # a larger set    -> unsure, flag
```

A singleton set means the model is confident enough to trust; a larger set is a
calibrated "not sure between these"; an empty set is "not confident about
anything", a principled abstention, not a coin flip.

## Why not just use the softmax?

Modern models are miscalibrated and usually overconfident: a "0.9" is not a 90%
chance of being right, and taking the top label gives you whatever the model's
accuracy happens to be, with no guarantee. Conformal prediction fixes the *output*
without retraining the model: from a small calibration set it computes a threshold
so that the returned set covers the truth at least at your target rate.

## Benchmark: the guarantee holds

`benchmarks/bench_coverage.py`: 5-class problem, 4,000 calibration / 4,000 test,
repeated over 50 independent calibration splits.

| Target coverage | Mean empirical | Per-run range | Avg set size |
|---|---|---|---|
| 99% | 99.0% | 98.2% to 99.4% | 4.01 |
| 95% | 95.0% | 93.9% to 96.0% | 2.79 |
| 90% | 90.0% | 88.0% to 91.6% | 1.91 |
| 80% | 80.1% | 78.1% to 81.9% | 1.13 |

![certain: conformal coverage tracks the target, naive top-1 does not](docs/benchmark.png)

The guarantee is **marginal**: coverage is at least `1 - alpha` in expectation over
the draw of the calibration set, not on every individual run. The mean column is
the quantity the theory bounds and it lands on the target; the range column shows
single runs falling either side, which is expected. A single split is not evidence
for or against the guarantee, so the benchmark averages rather than reporting one
seed. Regenerate the chart with `python docs/make_benchmark_chart.py`.

Naive top-1 coverage is just its accuracy, **79.1%, with no guarantee**.
Conformal coverage tracks the target you ask for, and the set size tells you how
much the model actually knows at that confidence level.

## Selective answering (abstain when unsure)

```python
from certain import SelectiveAnswerer

gate = SelectiveAnswerer(target_error=0.1)       # answer only if <=10% error
gate.calibrate(confidences, correct)             # (conf, was_correct) pairs

gate.answer(0.95)   # True  -> answer it
gate.answer(0.40)   # False -> abstain
gate.evaluate(confidences, correct)              # {'coverage': ..., 'error_on_answered': ...}
```

`SelectiveAnswerer` controls empirical error on answered items; for a strict
distribution-free bound use `ConformalClassifier`, which carries the finite-sample
coverage guarantee, or `ConformalRiskController` below, which bounds the loss you
actually care about.

## Drift: coverage that survives non-exchangeable traffic

Split conformal assumes exchangeability. Production violates it the moment
prompts, retrieval quality or user mix shift, and a threshold fitted once then
quietly stops covering at the rate it promised. `AdaptiveConformalClassifier`
nudges the working alpha from the misses it observes:
`alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`.

```python
from certain import AdaptiveConformalClassifier

clf = AdaptiveConformalClassifier(alpha=0.1, gamma=0.02)
clf.calibrate(cal_scores, cal_labels)

labels = clf.observe(scores, true_label)   # predict, then learn from the outcome
clf.error_rate                             # converges to alpha
```

Measured on a 5-class problem where the scoring model degrades hard partway
through (`tests/test_certain.py`), target coverage 90%:

| | Coverage after shift | Avg set size |
|---|---|---|
| `ConformalClassifier` (static) | **53.5%** | 1.12 |
| `AdaptiveConformalClassifier` | **89.9%** | 2.41 |

The static threshold loses a third of its coverage and the set size never admits
it. The adaptive one holds the target and pays in set size, which is visible.
Gibbs & Candes prove `|mean(err) - alpha| <= (alpha_1 + gamma) / (T * gamma)` for
any sequence, including adversarial, with no exchangeability assumption.

Scope: this adapts the working alpha, not the nonconformity distribution, which
stays as calibrated. It corrects the rate at which you miss. It cannot repair a
scoring function that has gone stale in a way the calibration set never saw.

## Bounding a loss that isn't miscoverage

"Is the true label in the set?" is one loss. Often the question is "how many of
these claims do I keep?" or "how many passages do I pass downstream?".
`ConformalRiskController` bounds **any** loss that shrinks as a parameter grows:

```python
from certain import ConformalRiskController

crc = ConformalRiskController(alpha=0.1)      # want expected loss <= 0.1
crc.calibrate(loss_rows, lambdas)             # loss_rows[i][j] = loss of item i at lambdas[j]
crc.lambda_                                   # smallest lambda that controls the risk
```

It fits `lambda_hat = inf{lambda : (n/(n+1)) * R_n(lambda) + B/(n+1) <= alpha}`,
giving `E[loss] <= alpha` on a fresh exchangeable item, finite-sample and
distribution-free. The miscoverage indicator recovers split conformal exactly.

The proof needs each item's loss to be non-increasing in lambda, so `calibrate`
checks that and raises rather than returning a number that looks calibrated and
is not.

## API

- **`ConformalClassifier(alpha=0.1)`**: `.calibrate(score_rows, labels)`,
  `.predict_set(scores)`, `.predict(scores)`, `.evaluate(score_rows, labels)`.
- **`AdaptiveConformalClassifier(alpha=0.1, gamma=0.01)`**: `.calibrate(...)`,
  `.predict_set(scores)`, `.record(covered)`, `.observe(scores, label)`,
  `.error_rate`, `.alpha_t`, `.evaluate(...)`.
- **`ConformalRiskController(alpha=0.1, loss_bound=1.0)`**:
  `.calibrate(loss_rows, lambdas)`, `.lambda_`, `.evaluate(...)`.
- **`SelectiveAnswerer(target_error=0.05)`**: `.calibrate(confidences, correct)`,
  `.answer(confidence)`, `.evaluate(...)`.

Scores are any non-negative per-label values (probabilities, exponentiated
logprobs, judge scores); they're normalised internally.

**Calibration size and small alpha.** A target of `1 - alpha` needs at least
`ceil(1/alpha) - 1` calibration examples before the conformal quantile is finite.
Below that, `qhat` saturates at 1.0 and every set contains every label: coverage is
trivially 1.0 and the set is useless. `alpha=0.01` therefore needs **>= 99**
calibration examples, `alpha=0.05` needs >= 19. The constructor accepts any alpha in
(0, 1), so this is on the caller to respect.

## Install

```bash
git clone https://github.com/ss1738/certain
cd certain && pip install -e .
```

Not published on PyPI; install from the clone. Python ≥ 3.9, no runtime
dependencies.

## References

Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*; Angelopoulos &
Bates, *A Gentle Introduction to Conformal Prediction* (2021); Gibbs & Candes,
*Adaptive Conformal Inference Under Distribution Shift*
([arXiv:2106.00170](https://arxiv.org/abs/2106.00170), NeurIPS 2021); Angelopoulos,
Bates, Fisch, Lei & Schuster, *Conformal Risk Control*
([arXiv:2208.02814](https://arxiv.org/abs/2208.02814)).

## Related

Part of a set of four small libraries for not trusting an unverified LLM output:

- [stigma](https://github.com/ss1738/stigma): aggregate many models with consensus that learns which to trust
- [disco](https://github.com/ss1738/disco): search with a verifier (an open FunSearch)
- [groundkit](https://github.com/ss1738/groundkit): ground a model's output against a source
- **certain** (this repo): calibrate confidence and abstain

## License

MIT © 2026 Satyawan Singh
