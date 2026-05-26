"""
Baseline comparison for model attribution (4-class LOO logistic regression).

Compares four feature sets:
  1. Shift vectors  (rewrite − original in function-word space) — our method
  2. Raw rewrite    (function-word vector of rewrite only, no subtraction)
  3. Original only  (function-word vector of original — sanity / lower bound)
  4. Char n-grams   (TF-IDF on char 3–6-grams of rewrite — Stamatatos 2013)
  5. Surface stats  (sentence length, TTR, punctuation density)
  6. Combined       (shift vectors + surface stats)
"""

from __future__ import annotations
import re
import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.stylometry import StyleAnalyzer, FUNCTION_WORDS_FR
from src.data import load_originals, load_llm_corpora


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def loo_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    """LOO logistic regression accuracy."""
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    )
    loo = LeaveOneOut()
    preds = []
    for train_idx, test_idx in loo.split(X):
        clf.fit(X[train_idx], y[train_idx])
        preds.append(clf.predict(X[test_idx])[0])
    return accuracy_score(y, preds)


def surface_features(texts: list[str]) -> np.ndarray:
    """Sentence length mean/std, type-token ratio, punctuation density."""
    rows = []
    for t in texts:
        sentences = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
        words = re.findall(r"\b\w+\b", t.lower())
        sent_lens = [len(re.findall(r"\b\w+\b", s)) for s in sentences] or [0]
        ttr = len(set(words)) / max(len(words), 1)
        punct = sum(1 for c in t if c in ".,;:!?\"'()—") / max(len(t), 1)
        rows.append([
            np.mean(sent_lens),
            np.std(sent_lens),
            ttr,
            punct,
            len(words),
        ])
    return np.array(rows)


def fw_vector(texts: list[str], sa: StyleAnalyzer) -> np.ndarray:
    """L2-normalised function-word frequency vectors."""
    return sa.fit_transform(texts)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    sa = StyleAnalyzer()
    originals   = load_originals()          # 80 texts
    llm_corpora = load_llm_corpora()        # {label: [80 rewrites]}

    labels_list = list(llm_corpora.keys())
    n_per_class = len(originals)

    # Build flat arrays
    orig_vecs = fw_vector(originals, sa)    # (80, 57)

    rewrites_flat: list[str] = []
    y: list[int] = []
    for cls_idx, label in enumerate(labels_list):
        rewrites_flat.extend(llm_corpora[label])
        y.extend([cls_idx] * n_per_class)
    y = np.array(y)                         # (320,)

    rew_vecs  = fw_vector(rewrites_flat, sa)  # (320, 57)

    # Tile orig_vecs to align with rewrites
    orig_tiled = np.tile(orig_vecs, (len(labels_list), 1))  # (320, 57)

    # ── Feature sets ──────────────────────────────────────────

    # 1. Shift vectors (our method)
    shift_vecs = rew_vecs - orig_tiled                       # (320, 57)

    # 2. Raw rewrite function-word vectors
    raw_rew = rew_vecs                                       # (320, 57)

    # 3. Original only (lower bound — should be ~25%)
    raw_orig = orig_tiled                                    # (320, 57)

    # 4. Char n-gram TF-IDF on rewrites (Stamatatos 2013)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6),
                            max_features=5000, sublinear_tf=True)
    char_ngrams = tfidf.fit_transform(rewrites_flat).toarray()  # (320, 5000)

    # 5. Surface statistics
    surf = surface_features(rewrites_flat)                   # (320, 5)

    # 6. Combined: shift + surface
    combined = np.hstack([shift_vecs, surf])                 # (320, 62)

    # ── Evaluate ─────────────────────────────────────────────

    experiments = [
        ("Shift vectors (ours)",         shift_vecs),
        ("Raw rewrite FW vectors",       raw_rew),
        ("Original FW vectors (sanity)", raw_orig),
        ("Char n-grams TF-IDF (3–6)",    char_ngrams),
        ("Surface statistics",           surf),
        ("Shift + surface (combined)",   combined),
    ]

    baseline = 1 / len(labels_list)
    print(f"\nBaseline (random): {baseline:.1%}\n")
    print(f"{'Method':<40} {'Accuracy':>10}")
    print("─" * 52)

    results = {}
    for name, X in experiments:
        acc = loo_accuracy(X, y)
        results[name] = acc
        marker = " ←" if "ours" in name else ""
        print(f"{name:<40} {acc:>9.1%}{marker}")

    print("─" * 52)

    # ── Key comparisons for paper ─────────────────────────────
    print("\nKey comparisons:")
    shift_acc = results["Shift vectors (ours)"]
    raw_acc   = results["Raw rewrite FW vectors"]
    orig_acc  = results["Original FW vectors (sanity)"]
    ngram_acc = results["Char n-grams TF-IDF (3–6)"]

    gain = shift_acc - raw_acc
    print(f"  Shift vs. raw rewrite: {shift_acc:.1%} vs {raw_acc:.1%} "
          f"({'+'if gain>=0 else ''}{gain:.1%} — "
          f"{'subtraction helps' if gain > 0.01 else 'subtraction neutral' if abs(gain) <= 0.01 else 'subtraction hurts'})")
    print(f"  Shift vs. char n-grams: {shift_acc:.1%} vs {ngram_acc:.1%}")
    print(f"  Original only (sanity): {orig_acc:.1%} "
          f"({'above' if orig_acc > baseline + 0.02 else 'near'} chance — "
          f"{'leak!' if orig_acc > baseline + 0.05 else 'ok'})")


if __name__ == "__main__":
    main()
