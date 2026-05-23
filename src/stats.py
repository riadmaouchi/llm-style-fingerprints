"""
src/stats.py
~~~~~~~~~~~~

Tests statistiques — délégués à stylometry.stats (source unique de vérité).

Ce module est conservé pour la compatibilité ascendante des notebooks et scripts.
Toutes les fonctions sont réexportées depuis stylometry-python >= 1.4.0.
"""

from stylometry.stats import (
    bootstrap_ci,
    compute_drift,
    detect_change_points,
    intra_variance,
    pairwise_tests,
    permutation_test,
)

__all__ = [
    "bootstrap_ci",
    "permutation_test",
    "pairwise_tests",
    "intra_variance",
    "detect_change_points",
    "compute_drift",
]
