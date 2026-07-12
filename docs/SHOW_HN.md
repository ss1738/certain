# Launch copy for certain

## Show HN title (pick one)
- **Show HN: certain, calibrated confidence and abstention for any model, in ~200 LOC**
- Show HN: Turn a softmax into a prediction set with a coverage guarantee
- Show HN: certain, conformal prediction + selective answering, zero dependencies

## Show HN body

A softmax "0.9" is not a 90% chance of being right, modern models are
miscalibrated and usually overconfident. Taking the top label gives you whatever
the model's accuracy happens to be, with no guarantee, which is a shaky thing to
build an automated decision on.

certain is a small, zero-dependency library that fixes the *output* without
touching the model. From a held-out calibration set it computes a threshold so
that:

- `ConformalClassifier` returns a **prediction set** guaranteed (distribution-free,
  finite-sample) to contain the true label with probability ≥ 1−α. A singleton set
  = confident, answer it; a bigger set = calibrated "unsure between these"; an
  empty set = principled abstention.
- `SelectiveAnswerer` turns a confidence signal into an **answer/abstain gate** that
  holds the error rate on answered items to a target.

It's model-agnostic, feed it LLM logprobs, judge scores, classifier probabilities,
retrieval scores.

Benchmark (5 classes, 4k cal / 4k test): naive top-1 coverage is just its accuracy,
**78.2% with no guarantee**. Conformal coverage tracks the target, 99%→98.6%,
95%→94.3%, 90%→88.6%, and the set size tells you how much the model actually knows
at that confidence.

Repo, tests, benchmark: https://github.com/ss1738/certain

Honest scope: `ConformalClassifier` carries the real finite-sample coverage
guarantee (assuming exchangeability). `SelectiveAnswerer` controls *empirical*
error and holds approximately on exchangeable test data; for a strict bound use the
conformal sets or add a Learn-then-Test correction. Feedback welcome.

## First comment

The underrated part is that conformal is a wrapper, not a model change, you keep
your classifier/LLM exactly as-is and only post-process its scores against a
calibration set. So the cost is one held-out split and a quantile, and in return
the *set* is honest even when the point prediction isn't. Empty and large sets are
features, not bugs: they're the model telling you, in calibrated terms, when to
route to a human.

## Part of a trio → quartet
certain is one of four small libraries on the theme "don't trust an unverified LLM
output":
- stigma, aggregate many models (consensus that learns which to trust)
- disco, search with a verifier (open FunSearch)
- groundkit, ground a single model's output (a model can propose, never assert)
- certain, calibrate & abstain (this one)

## Where else to post
- r/MachineLearning ("[P] ..."), r/statistics
- X/Twitter: the coverage table + "your softmax is not a probability"
- lobste.rs (ml / ai)
