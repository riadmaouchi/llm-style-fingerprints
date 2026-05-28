"""
Exploration de features pour améliorer la classification inter-LLM.
Teste : combinaisons de features × régularisation × augmentation P2.

Usage:
    python scripts/improve_baselines.py
"""
from __future__ import annotations
import re
import sys
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import load_llm_corpora, load_originals, load_single_model_aligned
from src.stylometry import StyleAnalyzer

# ─── Données P1 ──────────────────────────────────────────────────────────────

sa = StyleAnalyzer()
orig = load_originals()            # 80 originals
llm  = load_llm_corpora()          # {label: [80 rewrites]}
labels = list(llm.keys())          # 4 models

rew_flat: list[str] = []
y_all: list[int] = []
for ci, lab in enumerate(labels):
    rew_flat.extend(llm[lab])
    y_all.extend([ci] * len(llm[lab]))
y = np.array(y_all)                # (320,)

orig_vecs  = sa.fit_transform(orig)               # (80,41)
rew_vecs   = sa.fit_transform(rew_flat)           # (320,41)
orig_tiled = np.tile(orig_vecs, (len(labels), 1)) # (320,41)
shift_vecs = rew_vecs - orig_tiled                # (320,41)


# ─── Features ────────────────────────────────────────────────────────────────

def surface_features(texts: list[str]) -> np.ndarray:
    """7 surface statistics per text."""
    rows = []
    for t in texts:
        sents = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
        words = re.findall(r"\b\w+\b", t.lower())
        sl    = [len(re.findall(r"\b\w+\b", s)) for s in sents] or [0]
        rows.append([
            np.mean(sl),                                        # avg words/sent
            np.std(sl),                                         # std words/sent
            len(set(words)) / max(len(words), 1),               # TTR
            sum(c in ".,;:!?\"'()—" for c in t) / max(len(t), 1),  # punct density
            len(words),                                         # total words
            t.count(",")  / max(len(words), 1),                 # comma/word
            t.count(";")  / max(len(words), 1),                 # semicolon/word
        ])
    return np.array(rows)


surf = surface_features(rew_flat)

tfidf_36 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6),
                            max_features=5000, sublinear_tf=True)
char_feats = tfidf_36.fit_transform(rew_flat).toarray()  # (320, 5000)

# Word 2-grams
tfidf_w2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                            max_features=2000, sublinear_tf=True)
word_feats = tfidf_w2.fit_transform(rew_flat).toarray()


# ─── P2 augmentation ─────────────────────────────────────────────────────────

SLUGS = {"GPT-4": "gpt4", "Claude 3": "claude3", "Mistral 7B": "mistral", "Gemini Pro": "gemini"}

# Build per-model P2 arrays aligned by original index
p2_rew_by_model: dict[str, list[str]] = {}
p2_orig_by_model: dict[str, list[str]] = {}
for lab, slug in SLUGS.items():
    o2, r2 = load_single_model_aligned(slug, "p2")
    p2_rew_by_model[lab]  = r2
    p2_orig_by_model[lab] = o2


# ─── LOO helpers ─────────────────────────────────────────────────────────────

def loo_logreg(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> float:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    loo = LeaveOneOut()
    preds: list[int] = []
    for tr, te in loo.split(X):
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        preds.append(int(clf.predict(X[te])[0]))
    return accuracy_score(y, preds)


def loo_svm(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> float:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    loo = LeaveOneOut()
    preds: list[int] = []
    for tr, te in loo.split(X):
        clf = SVC(kernel="rbf", C=C, gamma="scale")
        clf.fit(X[tr], y[tr])
        preds.append(int(clf.predict(X[te])[0]))
    return accuracy_score(y, preds)


# ─── LOO avec augmentation P2 ────────────────────────────────────────────────

def loo_with_p2(X_p1: np.ndarray, y_p1: np.ndarray, C: float = 1.0) -> float:
    """LOO on P1 — but each training fold is augmented with P2 samples.
    For fold i (test = P1 sample i):
      - train on P1[~i] + P2 samples for originals != orig_i
    The P2 rewrite corresponding to orig_i is EXCLUDED to avoid leakage.
    n_per_class = 80.
    """
    n_cls = 80
    # Build P2 feature matrices per model (same feature pipeline as P1)
    p2_shift_by_model: dict[str, np.ndarray] = {}
    for lab, slug in SLUGS.items():
        r2 = p2_rew_by_model[lab]
        o2 = p2_orig_by_model[lab]
        rv = sa.fit_transform(r2)
        ov = sa.fit_transform(o2)
        p2_shift_by_model[lab] = rv - ov

    # Rebuild per-model shifts for P1 (already have shift_vecs but need per-model slices)
    p1_slices: list[np.ndarray] = []
    for ci in range(len(labels)):
        p1_slices.append(X_p1[ci * n_cls:(ci + 1) * n_cls])

    # Determine which columns of X_p1 correspond to shift_vecs
    # We use X_p1 as-is (the caller passes the feature matrix)
    # For P2 we need same features — this only works if features = shift_vecs
    # For simplicity: rebuild P2 features inline

    scaler = StandardScaler()
    loo = LeaveOneOut()
    preds: list[int] = []

    for fold_idx, (tr_mask, te_mask) in enumerate(loo.split(X_p1)):
        # Which original is being tested?
        test_global = te_mask[0]
        test_cls    = test_global // n_cls
        orig_idx    = test_global % n_cls   # index within the original 80

        # P1 training set
        X_tr = X_p1[tr_mask]
        y_tr = y_p1[tr_mask]

        # P2 training set: all models, exclude orig_idx
        X_p2_rows: list[np.ndarray] = []
        y_p2_rows: list[int] = []
        for ci, lab in enumerate(labels):
            mat = p2_shift_by_model[lab]   # (n_p2, 41)
            n_p2 = mat.shape[0]
            keep = [j for j in range(n_p2) if j != orig_idx]
            X_p2_rows.append(mat[keep])
            y_p2_rows.extend([ci] * len(keep))

        X_aug = np.vstack([X_tr] + X_p2_rows)
        y_aug = np.concatenate([y_tr, y_p2_rows])

        # Scale on augmented train, apply to test
        scaler.fit(X_aug)
        X_aug_s = scaler.transform(X_aug)
        X_te_s  = scaler.transform(X_p1[te_mask])

        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X_aug_s, y_aug)
        preds.append(int(clf.predict(X_te_s)[0]))

    return accuracy_score(y_p1, preds)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  FEATURE EXPLORATION — LLM attribution (n=320, baseline=25%)")
    print("=" * 65)

    # --- 1. Feature × C grid (LogReg) ---
    fw_pairs: list[tuple[str, np.ndarray]] = [
        ("shift",                shift_vecs),
        ("raw_fw",               rew_vecs),
        ("surface",              surf),
        ("shift + surface",      np.hstack([shift_vecs, surf])),
        ("shift + raw_fw",       np.hstack([shift_vecs, rew_vecs])),
        ("shift + raw_fw + surf",np.hstack([shift_vecs, rew_vecs, surf])),
        ("char_ngrams",          char_feats),
        ("shift + surf + char",  np.hstack([shift_vecs, surf, char_feats])),
        ("all",                  np.hstack([shift_vecs, rew_vecs, surf, char_feats])),
    ]

    print(f"\n{'Feature set':<30} {'C=0.1':>7} {'C=1':>7} {'C=5':>7}  best")
    print("-" * 65)
    best_name, best_X, best_acc, best_C = "", shift_vecs, 0.0, 1.0
    for name, X in fw_pairs:
        accs = [loo_logreg(X, y, C) for C in [0.1, 1.0, 5.0]]
        b = max(accs)
        bc = [0.1, 1.0, 5.0][accs.index(b)]
        print(f"{name:<30} {accs[0]:>6.1%} {accs[1]:>6.1%} {accs[2]:>6.1%}  {b:.1%}")
        if b > best_acc:
            best_acc, best_name, best_X, best_C = b, name, X, bc
    print("-" * 65)
    print(f"Best (LogReg): {best_name} @ C={best_C} → {best_acc:.1%}")

    # --- 2. SVM RBF on top candidates ---
    print(f"\n{'Feature set (SVM RBF)':<30} {'C=1':>7} {'C=5':>7}")
    print("-" * 50)
    for name, X in [
        ("shift + surface",      np.hstack([shift_vecs, surf])),
        ("shift + raw_fw + surf",np.hstack([shift_vecs, rew_vecs, surf])),
        ("shift + surf + char",  np.hstack([shift_vecs, surf, char_feats])),
    ]:
        a1, a5 = loo_svm(X, y, 1.0), loo_svm(X, y, 5.0)
        print(f"{name:<30} {a1:>6.1%} {a5:>6.1%}")

    # --- 3. P2 augmentation ---
    print(f"\n{'P2 augmentation (shift, LogReg)':<40} {'C=0.5':>7} {'C=1':>7}")
    print("-" * 55)
    for C in [0.5, 1.0]:
        acc = loo_with_p2(shift_vecs, y, C)
        print(f"  shift + P2 aug  C={C:<4}                     {acc:>6.1%}")
    for C in [0.5, 1.0]:
        acc = loo_with_p2(np.hstack([shift_vecs, surf]), y, C)
        print(f"  shift+surf+P2   C={C:<4}                     {acc:>6.1%}")

    print()


if __name__ == "__main__":
    main()
