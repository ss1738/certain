"""certain -- calibrated confidence and abstention for model outputs.

Two tools, model-agnostic and dependency-free. :class:`ConformalClassifier`
turns per-label scores into prediction *sets* with a finite-sample coverage
guarantee (the true label is in the set with probability >= 1 - alpha).
:class:`SelectiveAnswerer` turns a confidence signal into an answer/abstain gate
that holds the error rate on answered items to a target. Both work over any
scores -- LLM logprobs, judge scores, classifier probabilities, retrieval scores.
"""

from .classifier import ConformalClassifier
from .selective import SelectiveAnswerer

__version__ = "0.1.0"
__all__ = ["ConformalClassifier", "SelectiveAnswerer", "__version__"]
