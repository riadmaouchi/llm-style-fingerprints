"""
src/features.py
~~~~~~~~~~~~~~~
Features stylistiques complémentaires aux mots-outils de la librairie.
"""
from __future__ import annotations

import math
import re

import numpy as np

HEDGE_WORDS_FR: list[str] = [
    "toutefois", "néanmoins", "cependant", "pourtant",
    "notamment", "effectivement", "certes", "bien entendu",
    "il convient", "en effet", "ainsi", "donc",
    "par ailleurs", "de plus", "en outre", "afin de",
    "il semble", "il paraît", "peut-être", "probablement",
    "sans doute", "vraisemblablement", "apparemment",
    "en quelque sorte", "à vrai dire",
]


def hedge_density(text: str) -> float:
    """Fréquence des hedge words pour 100 tokens."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    n = len(tokens)
    if n == 0:
        return 0.0
    count = sum(text.lower().count(h) for h in HEDGE_WORDS_FR)
    return count / n * 100.0


def burstiness(text: str) -> float:
    """Coefficient de variation de la longueur des phrases (std / mean)."""
    sentences = re.split(r"[.!?]+", text)
    lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.0
    m = float(np.mean(lengths))
    if m == 0.0:
        return 0.0
    return float(np.std(lengths) / m)


def punctuation_entropy(text: str) -> float:
    """Entropie de Shannon sur la distribution des signes de ponctuation."""
    punct_chars = re.findall(r"""[.,;:!?"'()\[\]—–\-]""", text)
    if not punct_chars:
        return 0.0
    total = len(punct_chars)
    counts: dict[str, int] = {}
    for c in punct_chars:
        counts[c] = counts.get(c, 0) + 1
    return float(-sum((v / total) * math.log2(v / total) for v in counts.values()))


def extra_feature_names() -> list[str]:
    return ["hedge_density", "burstiness", "punct_entropy"]


def extra_features(texts: list[str]) -> np.ndarray:
    """
    Retourne (n, 3) [hedge_density, burstiness, punct_entropy].
    Z-scoré globalement sur le batch fourni — appeler avec tous les textes
    des groupes combinés pour que la normalisation soit partagée.
    """
    raw = np.array([
        [hedge_density(t), burstiness(t), punctuation_entropy(t)]
        for t in texts
    ], dtype=float)
    mu = raw.mean(axis=0)
    sigma = raw.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    return (raw - mu) / sigma


__all__ = [
    "HEDGE_WORDS_FR",
    "hedge_density",
    "burstiness",
    "punctuation_entropy",
    "extra_feature_names",
    "extra_features",
]
