"""certain -- calibrated confidence and abstention for model outputs.

Four tools, model-agnostic and dependency-free.

:class:`ConformalClassifier` turns per-label scores into prediction *sets* with a
finite-sample coverage guarantee (the true label is in the set with probability
>= 1 - alpha). :class:`AdaptiveConformalClassifier` keeps that coverage when the
data stops being exchangeable, by adapting the working alpha to the misses it
observes. :class:`ConformalRiskController` generalises the guarantee from
miscoverage to any monotone loss, which is what you need for "how many claims do
I keep?" rather than "which label is it?". :class:`SelectiveAnswerer` is the
simple answer/abstain gate, controlling empirical error only.

All of them work over any scores: LLM logprobs, judge scores, classifier
probabilities, retrieval scores.
"""

from .adaptive import AdaptiveConformalClassifier
from .classifier import ConformalClassifier
from .risk import ConformalRiskController
from .selective import SelectiveAnswerer

__version__ = "0.2.0"
__all__ = [
    "AdaptiveConformalClassifier",
    "ConformalClassifier",
    "ConformalRiskController",
    "SelectiveAnswerer",
    "__version__",
]
