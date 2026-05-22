"""
src/stats.py
~~~~~~~~~~~~

Tests statistiques pour valider les résultats stylométriques.

Fonctions exportées
-------------------
bootstrap_ci()    : intervalle de confiance percentile par bootstrap
permutation_test(): test de permutation bilatéral (H₀ : mêmes distributions)
pairwise_tests()  : tests de Welch avec correction multiple (Bonferroni / Holm)
intra_variance()  : distance cosinus intra-groupe (baseline bruit naturel)
"""

from __future__ import annotations

import itertools
from itertools import combinations
from typing import Literal

import numpy as np
from scipy.stats import ttest_ind

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: list[float] | np.ndarray,
    n_boot: int = 5_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Intervalle de confiance Bootstrap percentile.

    Paramètres
    ----------
    data   : échantillon de shifts (floats)
    n_boot : nombre de tirages bootstrap
    ci     : niveau de confiance (0.95 → IC 95 %)
    rng    : générateur NumPy pour la reproductibilité

    Retourne
    --------
    (lower_bound, upper_bound)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(data, dtype=float)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    group_a: list[float] | np.ndarray,
    group_b: list[float] | np.ndarray,
    n_perm: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Test de permutation bilatéral sur la différence de moyennes.

    H₀  : les deux groupes ont la même distribution de shifts.
    Stat : |mean(A) − mean(B)|

    Retourne
    --------
    dict avec clés :
        observed          — statistique observée
        p_value           — p-value (borne inférieure conservative : 1/n_perm)
        null_distribution — array des statistiques sous H₀
        significant_05    — bool (p < 0.05)
        significant_01    — bool (p < 0.01)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    observed = float(abs(a.mean() - b.mean()))
    n_a = len(a)

    combined = np.concatenate([a, b])
    perms = rng.permutation(
        np.tile(combined, (n_perm, 1)).T
    ).T                                      # shape (n_perm, n_a + n_b)
    null = np.abs(perms[:, :n_a].mean(axis=1) - perms[:, n_a:].mean(axis=1))

    p_value = float(max((null >= observed).mean(), 1.0 / n_perm))
    return {
        "observed": observed,
        "p_value": p_value,
        "null_distribution": null,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
    }


# ---------------------------------------------------------------------------
# Pairwise tests
# ---------------------------------------------------------------------------

_CORRECTIONS = frozenset({"bonferroni", "holm", "none"})


def pairwise_tests(
    shifts_dict: dict[str, list[float]],
    correction: Literal["bonferroni", "holm", "none"] = "bonferroni",
) -> list[dict]:
    """
    Tests de Welch sur toutes les paires avec correction multiple.

    Paramètres
    ----------
    shifts_dict : {"GPT-4": [...], "Claude 3": [...], ...}
    correction  : "bonferroni" | "holm" | "none"

    Retourne
    --------
    liste de dicts triée par p_corrected croissant
    """
    if correction not in _CORRECTIONS:
        raise ValueError(f"correction must be one of {_CORRECTIONS!r}, got {correction!r}")

    models = list(shifts_dict.keys())
    pairs  = list(itertools.combinations(models, 2))
    n_tests = len(pairs)

    raw: list[dict] = []
    for a, b in pairs:
        t, p = ttest_ind(shifts_dict[a], shifts_dict[b], equal_var=False)
        raw.append({"model_a": a, "model_b": b, "t": float(t), "p_raw": float(p)})

    raw.sort(key=lambda r: r["p_raw"])

    for i, r in enumerate(raw):
        if correction == "bonferroni":
            p_corr = min(r["p_raw"] * n_tests, 1.0)
        elif correction == "holm":
            p_corr = min(r["p_raw"] * (n_tests - i), 1.0)
        else:
            p_corr = r["p_raw"]
        r.update({
            "p_corrected": p_corr,
            "significant_05": p_corr < 0.05,
            "correction": correction,
            "n_tests": n_tests,
        })

    return sorted(raw, key=lambda r: r["p_corrected"])


# ---------------------------------------------------------------------------
# Intra-group variance
# ---------------------------------------------------------------------------

def intra_variance(
    texts_dict: dict[str, list[str]],
    analyzer,
) -> dict[str, float]:
    """
    Distance cosinus intra-groupe : bruit naturel de style d'un groupe.

    Pour chaque groupe : moyenne des distances cosinus entre toutes les paires.
    Un groupe homogène (style LLM convergent) a une variance intra faible.

    Paramètres
    ----------
    texts_dict : {"Zola": [...], "GPT-4": [...], ...}
    analyzer   : instance de StyleAnalyzer (doit exposer shift(t1, t2))

    Retourne
    --------
    {"Zola": 0.48, "GPT-4": 0.54, ...}
    """
    result: dict[str, float] = {}
    for label, texts in texts_dict.items():
        if len(texts) < 2:
            result[label] = 0.0
            continue
        dists = [analyzer.shift(t1, t2) for t1, t2 in combinations(texts, 2)]
        result[label] = float(np.mean(dists))
    return result
