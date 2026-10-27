"""certain -- calibrated confidence and abstention for model outputs.

Model-agnostic and dependency-free. Everything works over any scores: LLM
logprobs, judge scores, classifier probabilities, retrieval scores.

:class:`ConformalClassifier` turns per-label scores into prediction *sets* with a
finite-sample coverage guarantee (the true label is in the set with probability
>= 1 - alpha). :class:`MondrianConformalClassifier` makes that guarantee hold
within each group you declare, rather than only on average, because a marginal
number can hide a group being covered half as well as the headline claims.
:class:`AdaptiveConformalClassifier` keeps coverage when the data stops being
exchangeable, by adapting the working alpha to the misses it observes.
:class:`ConformalRiskController` generalises the guarantee from miscoverage to
any monotone loss, which is what "how many claims do I keep?" needs.
:class:`SelectiveAnswerer` is the simple answer/abstain gate, controlling
empirical error only.

:func:`training_conditional_bound` says what any one calibration set is actually
worth, which matters most exactly where the set is small.
"""

from .adaptive import AdaptiveConformalClassifier
from .bounds import training_conditional_bound
from .classifier import ConformalClassifier
from .mondrian import MondrianConformalClassifier
from .risk import ConformalRiskController
from .selective import SelectiveAnswerer

__version__ = "0.3.0"
__all__ = [
    "AdaptiveConformalClassifier",
    "ConformalClassifier",
    "ConformalRiskController",
    "MondrianConformalClassifier",
    "SelectiveAnswerer",
    "__version__",
    "training_conditional_bound",
]
