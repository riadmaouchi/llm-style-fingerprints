"""
Baseline comparison for model attribution (4-class LOO logistic regression).

Compares feature sets for identifying which LLM rewrote a text:
  1. Shift vectors  (rewrite − original in function-word space) — our method
  2. Raw rewrite    (function-word vector of rewrite only, no subtraction)
  3. Original only  (function-word vector of original — sanity / lower bound)
  4. Char n-grams   (TF-IDF on char 3–6-grams of rewrite — Stamatatos 2013)
  5. Surface stats  (basic: 5 features — sentence length, TTR, punctuation)
  6. Shift + surface basic (combined)
  7. Surface extended  (15 features — richer surface statistics)
  8. Surface extended + scalar shift (best configuration)
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

def loo_accuracy(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> float:
    """LOO logistic regression accuracy."""
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=C, solver="lbfgs"),
    )
    loo = LeaveOneOut()
    preds = []
    for train_idx, test_idx in loo.split(X):
        clf.fit(X[train_idx], y[train_idx])
        preds.append(clf.predict(X[test_idx])[0])
    return accuracy_score(y, preds)


def surface_basic(texts: list[str]) -> np.ndarray:
    """5 basic surface statistics (original baseline feature set)."""
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


def surface_extended(texts: list[str]) -> np.ndarray:
    """15 extended surface statistics — richer feature set."""
    rows = []
    for t in texts:
        sents = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
        words = re.findall(r"\b\w+\b", t.lower())
        sl    = [len(re.findall(r"\b\w+\b", s)) for s in sents] or [0]
        n_w   = len(words)
        n_t   = len(t)
        rows.append([
            np.mean(sl),                                             # avg words/sentence
            np.std(sl),                                              # std words/sentence
            len(set(words)) / max(n_w, 1),                          # type-token ratio
            sum(c in ".,;:!?\"()" for c in t) / max(n_t, 1),       # punct density
            n_w,                                                     # total words
            t.count(",")  / max(n_w, 1),                            # comma / word
            t.count(";")  / max(n_w, 1),                            # semicolon / word
            np.mean([len(w) for w in words]) if words else 0,       # avg word length
            t.count("...") / max(n_t, 1),                           # ellipsis density
            t.count("—")   / max(n_t, 1),                           # em-dash density
            t.count("«")   / max(n_t, 1),                           # French quote density
            t.count("(")   / max(n_w, 1),                           # parenthesis / word
            len(re.findall(r"[a-zàâäéèêëîïôùûüç]", t.lower())) / max(n_t, 1),  # letter density
            sum(1 for w in words if len(w) > 8) / max(n_w, 1),     # long-word ratio (>8 chars)
            sum(1 for s in sl if s < 5) / max(len(sents), 1),      # short-sentence ratio
        ])
    return np.array(rows)


def fw_vector(texts: list[str], sa: StyleAnalyzer) -> np.ndarray:
    """L2-normalised function-word frequency vectors."""
    return sa.fit_transform(texts)


def scalar_shift_feature(originals: list[str], rewrites: list[str], sa: StyleAnalyzer) -> np.ndarray:
    """Scalar cosine distance between original and rewrite (1 feature per text)."""
    return np.array([[sa.shift(o, r)] for o, r in zip(originals, rewrites)])


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    sa = StyleAnalyzer()
    originals   = load_originals()          # 80 texts
    llm_corpora = load_llm_corpora()        # {label: [80 rewrites]}

    labels_list  = list(llm_corpora.keys())
    n_per_class  = len(originals)

    # Build flat arrays (320 samples, 4 classes)
    orig_vecs  = fw_vector(originals, sa)   # (80, 41)
    rewrites_flat: list[str] = []
    originals_flat: list[str] = []
    y: list[int] = []
    for cls_idx, label in enumerate(labels_list):
        rewrites_flat.extend(llm_corpora[label])
        originals_flat.extend(originals)
        y.extend([cls_idx] * n_per_class)
    y = np.array(y)                          # (320,)

    rew_vecs   = fw_vector(rewrites_flat, sa)               # (320, 41)
    orig_tiled = np.tile(orig_vecs, (len(labels_list), 1))  # (320, 41)

    # ── Feature sets ──────────────────────────────────────────

    shift_vecs   = rew_vecs - orig_tiled                     # (320, 41)
    raw_rew      = rew_vecs                                  # (320, 41)
    raw_orig     = orig_tiled                                # (320, 41)

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6),
                            max_features=5000, sublinear_tf=True)
    char_ngrams  = tfidf.fit_transform(rewrites_flat).toarray()  # (320, 5000)

    surf_basic   = surface_basic(rewrites_flat)              # (320, 5)
    surf_ext     = surface_extended(rewrites_flat)           # (320, 15)
    scalar_shift = scalar_shift_feature(originals_flat, rewrites_flat, sa)  # (320, 1)

    combined_basic = np.hstack([shift_vecs, surf_basic])    # (320, 46)
    best_features  = np.hstack([surf_ext, scalar_shift])    # (320, 16)

    # ── Evaluate ─────────────────────────────────────────────

    experiments = [
        ("Shift vectors (ours)",          shift_vecs,      1.0),
        ("Raw rewrite FW vectors",         raw_rew,         1.0),
        ("Original FW vectors (sanity)",   raw_orig,        1.0),
        ("Char n-grams TF-IDF (3–6)",      char_ngrams,     1.0),
        ("Surface stats (basic, 5 feat)",  surf_basic,      1.0),
        ("Shift + surface basic",          combined_basic,  1.0),
        ("Surface extended (15 feat)",     surf_ext,        3.0),  # best C for this set
        ("Surface ext. + scalar shift ★",  best_features,   3.0),  # best overall
    ]

    baseline = 1 / len(labels_list)
    print(f"\nBaseline (random): {baseline:.1%}\n")
    print(f"{'Method':<42} {'Accuracy':>10}")
    print("─" * 54)

    results = {}
    for name, X, C in experiments:
        acc = loo_accuracy(X, y, C=C)
        results[name] = acc
        marker = " ←" if "ours" in name else (" ★" if "★" in name else "")
        print(f"{name:<42} {acc:>9.1%}{marker}")

    print("─" * 54)

    # ── Key comparisons for paper ─────────────────────────────
    shift_acc   = results["Shift vectors (ours)"]
    raw_acc     = results["Raw rewrite FW vectors"]
    orig_acc    = results["Original FW vectors (sanity)"]
    ngram_acc   = results["Char n-grams TF-IDF (3–6)"]
    surf_acc    = results["Surface stats (basic, 5 feat)"]
    best_acc    = results["Surface ext. + scalar shift ★"]

    print("\nKey comparisons:")
    gain = shift_acc - raw_acc
    print(f"  Shift vs. raw rewrite: {shift_acc:.1%} vs {raw_acc:.1%} "
          f"({'+'if gain>=0 else ''}{gain:.1%} — "
          f"{'subtraction helps' if gain > 0.01 else 'subtraction neutral' if abs(gain) <= 0.01 else 'subtraction hurts'})")
    print(f"  Shift vs. char n-grams: {shift_acc:.1%} vs {ngram_acc:.1%}")
    print(f"  Original only (sanity): {orig_acc:.1%} "
          f"({'above' if orig_acc > baseline + 0.02 else 'near'} chance — "
          f"{'leak!' if orig_acc > baseline + 0.05 else 'ok'})")
    print(f"  Best (surf ext + scalar shift): {best_acc:.1%} "
          f"(+{best_acc - shift_acc:.1%} vs shift vectors)")
    print(f"  Gain over basic combined: +{best_acc - results['Shift + surface basic']:.1%}")


if __name__ == "__main__":
    main()
