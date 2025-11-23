"""
N-gram language models for spelling correction.
"""

from .ngram_model import (
    NGramModel,
    SpellingChecker,
    EditDistance,
    Evaluator,
    CorrectionResult,
    EvaluationMetrics,
    load_training_corpus,
    load_english_dictionary,
)

__all__ = [
    'NGramModel',
    'SpellingChecker',
    'EditDistance',
    'Evaluator',
    'CorrectionResult',
    'EvaluationMetrics',
    'load_training_corpus',
    'load_english_dictionary',
]


