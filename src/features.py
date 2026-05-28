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


# ─── Extended surface statistics (nouvelle méthode, meilleure classification) ─

SURFACE_EXTENDED_NAMES: list[str] = [
    "sent_len_mean",     # avg words per sentence
    "sent_len_std",      # sentence length variability
    "ttr",               # type-token ratio
    "punct_density",     # punctuation chars / text length
    "word_count",        # total words
    "comma_per_word",    # comma / word
    "semicolon_per_word",# semicolon / word
    "avg_word_len",      # avg character length of words
    "ellipsis_density",  # ellipsis / text length
    "emdash_density",    # em-dash / text length
    "fr_quote_density",  # French « / text length
    "paren_per_word",    # parenthesis / word
    "letter_density",    # letter chars / text length
    "long_word_ratio",   # words >8 chars / all words
    "short_sent_ratio",  # sentences <5 words / all sentences
]


def surface_extended(texts: list[str]) -> np.ndarray:
    """
    15 surface statistics per text — best feature set for LLM attribution
    (LOO logistic regression accuracy: 56.2% standalone, 57.5% with scalar shift).
    """
    rows = []
    for t in texts:
        sents = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
        words = re.findall(r"\b\w+\b", t.lower())
        sl    = [len(re.findall(r"\b\w+\b", s)) for s in sents] or [0]
        n_w   = len(words)
        n_t   = len(t)
        rows.append([
            float(np.mean(sl)),
            float(np.std(sl)),
            len(set(words)) / max(n_w, 1),
            sum(c in ".,;:!?\"()" for c in t) / max(n_t, 1),
            float(n_w),
            t.count(",")   / max(n_w, 1),
            t.count(";")   / max(n_w, 1),
            float(np.mean([len(w) for w in words])) if words else 0.0,
            t.count("...") / max(n_t, 1),
            t.count("—")   / max(n_t, 1),
            t.count("«")   / max(n_t, 1),
            t.count("(")   / max(n_w, 1),
            len(re.findall(r"[a-zàâäéèêëîïôùûüç]", t.lower())) / max(n_t, 1),
            sum(1 for w in words if len(w) > 8) / max(n_w, 1),
            sum(1 for s in sl if s < 5) / max(len(sents), 1),
        ])
    return np.array(rows, dtype=float)


__all__ = [
    "HEDGE_WORDS_FR",
    "SURFACE_EXTENDED_NAMES",
    "hedge_density",
    "burstiness",
    "punctuation_entropy",
    "extra_feature_names",
    "extra_features",
    "surface_extended",
]
