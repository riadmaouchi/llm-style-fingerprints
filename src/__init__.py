# llm-style-fingerprints — public API
from src.data import load_corpus, load_llm_corpora, load_originals
from src.stats import bootstrap_ci, intra_variance, pairwise_tests, permutation_test
from src.stylometry import FUNCTION_WORDS_FR, PALETTE, StyleAnalyzer

__all__ = [
    "StyleAnalyzer",
    "PALETTE",
    "FUNCTION_WORDS_FR",
    "load_corpus",
    "load_originals",
    "load_llm_corpora",
    "bootstrap_ci",
    "permutation_test",
    "pairwise_tests",
    "intra_variance",
]
