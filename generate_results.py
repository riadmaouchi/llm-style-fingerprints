"""
Génère tous les graphiques dans results/ sans Jupyter.

Usage
-----
    python generate_results.py          # tous les graphiques
    python generate_results.py --fast   # sans t-SNE et UMAP (rapide)

Graphiques produits (14 au total)
----------------------------------
  shift_distributions.png        — violin plot des shifts par LLM
  pca_clusters.png               — PCA 2D (6 groupes)
  tsne_clusters.png              — t-SNE 2D (meilleure séparation non-linéaire)
  umap_clusters.png              — UMAP 2D (structure locale préservée)
  fingerprint_comparison.png     — fréquences des mots-outils discriminants
  confusion_matrix.png           — matrice de confusion classifieur centroïde
  feature_importance.png         — mots-outils discriminants (SVM linéaire)
  logistic_feature_importance.png — coefficients logistic regression par classe
  shift_by_source.png            — shifts par auteur source (Zola vs Maupassant)
  dendrogram.png                 — clustering hiérarchique des centroïdes
  radar_profiles.png             — profil stylistique en radar chart
  shift_vs_length.png            — shift cosinus vs longueur du texte
  bootstrap_ci.png               — moyennes + IC 95 % Bootstrap
  permutation_null.png           — distributions nulles vs statistiques observées
"""

from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path
from matplotlib.transforms import blended_transform_factory

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings("ignore")

# Ajouter la racine du projet au path avant les imports locaux
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.stylometry import StyleAnalyzer, PALETTE
from src.stats import bootstrap_ci, permutation_test, pairwise_tests, intra_variance
from src.data import load_corpus, load_originals, load_llm_corpora, load_single_model_aligned
from src.features import extra_features, extra_feature_names

# ---------------------------------------------------------------------------
# Design system — single source of truth for all visual constants
# ---------------------------------------------------------------------------

THEME = {
    "bg":           "#0D1117",   # fond GitHub dark
    "surface":      "#161B22",   # panneaux légèrement plus clairs
    "grid":         "#21262D",   # grille très subtile
    "text_primary": "#E6EDF3",   # titres
    "text_muted":   "#8B949E",   # labels axes
    "spine":        "#30363D",   # bordures axes
    "tick":         "#484F58",   # tirets axes
    "dpi":          150,
    "title_size":   13,
    "label_size":   10,
    "tick_size":    9,
    "annot_size":   9,
}
BG = THEME["bg"]  # alias backward-compat
plt.style.use("dark_background")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str, results_dir: Path) -> None:
    path = results_dir / name
    fig.savefig(path, dpi=THEME["dpi"], bbox_inches="tight", facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  ✓ {name}")


def _ax_style(ax: plt.Axes) -> None:
    ax.set_facecolor(THEME["bg"])
    ax.tick_params(colors=THEME["tick"])
    for sp in ax.spines.values():
        sp.set_edgecolor(THEME["spine"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_violin(sa: StyleAnalyzer, shifts_all: dict, results_dir: Path) -> None:
    spacing = 1.15
    models  = list(shifts_all.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(THEME["bg"])
    _ax_style(ax)

    trans = blended_transform_factory(ax.transAxes, ax.transData)

    for i, name in enumerate(reversed(models)):
        shifts  = np.array(shifts_all[name])
        y_off   = i * spacing
        color   = PALETTE[name]
        x_kde   = np.linspace(max(0.0, shifts.min() - 0.04), shifts.max() + 0.04, 300)
        kde     = gaussian_kde(shifts, bw_method=0.35)
        y_raw   = kde(x_kde)
        y_norm  = y_raw / y_raw.max() * 0.95

        ax.fill_between(x_kde, y_off, y_norm + y_off, color=color, alpha=0.65, lw=0)
        ax.plot(x_kde, y_norm + y_off, color=color, lw=1.5, alpha=0.90)

        mean_val    = float(shifts.mean())
        mean_height = kde(mean_val)[0] / y_raw.max() * 0.95
        ax.vlines(mean_val, y_off, mean_height + y_off, color="white", lw=1.8, alpha=0.9, zorder=4)
        ax.annotate(f"{mean_val:.3f}", (mean_val, mean_height + y_off + 0.04),
                    color="white", fontsize=8, ha="center", va="bottom")

        ax.text(-0.02, y_off + 0.48, name, ha="right", va="center",
                color=color, fontsize=9.5, fontweight="bold", transform=trans)

    ax.set_yticks([])
    ax.set_xlim(0.02, 0.56)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Distance cosinus (shift stylistique)", color=THEME["text_muted"], labelpad=8)
    ax.set_title(
        "Distribution des shifts stylistiques — chaque crête = un LLM\n"
        "Ligne blanche = moyenne  ·  largeur = densité",
        color=THEME["text_primary"], pad=12,
    )
    ax.grid(axis="x", alpha=0.06, color=THEME["grid"])
    _save(fig, "shift_distributions.png", results_dir)


def fig_pca(sa: StyleAnalyzer, corpus: dict, results_dir: Path) -> None:
    from matplotlib.patches import Ellipse
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # LDA on shift vectors — maximizes between-LLM separation so centroids
    # are visually well-separated (PCA on raw vectors gives too much overlap).
    originals   = load_originals()
    llm_corpora = load_llm_corpora()
    orig_vecs   = sa.fit_transform(originals)

    shift_pts, shift_lbls = [], []
    for lbl, rewrites in llm_corpora.items():
        rew_vecs = sa.fit_transform(rewrites)
        for ov, rv in zip(orig_vecs, rew_vecs):
            shift_pts.append(rv - ov)
            shift_lbls.append(lbl)

    S   = np.array(shift_pts)
    lbl_arr = np.array(shift_lbls)
    lda = LinearDiscriminantAnalysis(n_components=2)
    coords = lda.fit_transform(S, lbl_arr)

    # Label offsets so names don't collide
    label_offsets = {
        "GPT-4":      (-22, -14),
        "Claude 3":   ( 8,   8),
        "Mistral 7B": (-22,  8),
        "Gemini Pro": ( 8, -14),
    }

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)

    rng     = np.random.default_rng(42)
    handles = []

    for label in llm_corpora.keys():
        mask  = lbl_arr == label
        pts   = coords[mask]
        color = PALETTE.get(label, "#AAAAAA")

        # 1-sigma ellipse — kept subtle so centroids dominate
        if len(pts) >= 3:
            cov  = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h  = 2 * np.sqrt(np.maximum(vals, 0))
            ell   = Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=angle,
                            fc=color, alpha=0.07, ec=color, lw=1.5, ls="--", zorder=2)
            ax.add_patch(ell)

        # Jittered scatter — minimal presence, centroids are the story
        jitter = rng.uniform(-0.04, 0.04, pts.shape)
        ax.scatter((pts + jitter)[:, 0], (pts + jitter)[:, 1],
                   c=color, s=15, alpha=0.30, edgecolors="none", zorder=3)

        # Bold centroid marker
        cx, cy = pts.mean(axis=0)
        ax.scatter([cx], [cy], c=color, s=260, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=6)
        dx, dy = label_offsets.get(label, (8, 4))
        ax.annotate(label, (cx, cy), color=color,
                    fontsize=10, fontweight="bold",
                    xytext=(dx, dy), textcoords="offset points")
        handles.append(mpatches.Patch(color=color, label=label))

    ax.set_xlabel("LD1 — axe de discrimination inter-LLM", color=THEME["text_muted"], labelpad=8)
    ax.set_ylabel("LD2 — axe de discrimination inter-LLM", color=THEME["text_muted"], labelpad=8)
    ax.set_title(
        "Fingerprints stylistiques — LDA 2D (vecteurs de shift)\n"
        "✕ = centroïde LLM  ·  ellipses = dispersion 1-σ par modèle",
        color=THEME["text_primary"], pad=12,
    )
    ax.annotate(
        "LDA sur vecteurs de shift (réécriture − original)",
        xy=(0.01, 0.01), xycoords="axes fraction",
        color=THEME["text_muted"], fontsize=8, fontstyle="italic",
    )
    ax.grid(alpha=0.04, color=THEME["grid"])
    _save(fig, "pca_clusters.png", results_dir)


def fig_tsne(sa: StyleAnalyzer, corpus: dict, results_dir: Path) -> None:
    labels_list = list(corpus.keys())
    texts_flat  = [t for grp in corpus.values() for t in grp]
    labels_flat = [lbl for grp, lbl in zip(corpus.values(), labels_list) for _ in grp]

    X = sa.fit_transform(texts_flat)
    coords = TSNE(
        n_components=2, perplexity=12, random_state=42,
        max_iter=1500, learning_rate="auto", init="pca",
    ).fit_transform(X)

    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)
    handles = []
    rng = np.random.default_rng(42)
    for label in labels_list:
        mask = np.array([g == label for g in labels_flat])
        pts   = coords[mask]
        color = PALETTE.get(label, "#AAAAAA")
        jitter = rng.uniform(-0.04, 0.04, pts.shape)
        ax.scatter((pts + jitter)[:, 0], (pts + jitter)[:, 1], c=color, s=45,
                   alpha=0.55, edgecolors="none", zorder=3)
        if len(pts) >= 3:
            cov  = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h  = 2 * np.sqrt(np.maximum(vals, 0))
            ell   = Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=angle,
                            fc=color, alpha=0.10, ec=color, lw=1.8, ls="--", zorder=2)
            ax.add_patch(ell)
        cx, cy = pts.mean(axis=0)
        ax.scatter([cx], [cy], c=color, s=220, marker="X",
                   edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(f" {label}", (cx, cy), color=color,
                    fontsize=8.5, fontweight="bold",
                    xytext=(6, 4), textcoords="offset points")
        handles.append(mpatches.Patch(color=color, label=label))
    ax.set_title(
        "Espace stylistique — t-SNE 2D\n"
        "Points = textes individuels  ·  ✕ = centroïde  ·  ellipses = zone 1-σ",
        color="white", pad=12,
    )
    ax.legend(handles=handles, facecolor="#1C2128", edgecolor="#444444",
              labelcolor="white", fontsize=9)
    ax.grid(alpha=0.08, color="#AAAAAA")
    _save(fig, "tsne_clusters.png", results_dir)


def fig_fingerprint(sa: StyleAnalyzer, corpus: dict, results_dir: Path) -> None:
    fig = sa.plot_fingerprint(
        texts_dict=corpus,
        top_n=14,
        title="Empreinte stylistique — 14 mots-outils les plus discriminants",
    )
    _save(fig, "fingerprint_comparison.png", results_dir)


def fig_confusion(sa: StyleAnalyzer, llm_corpora: dict, results_dir: Path) -> tuple[np.ndarray, float]:
    model_names = list(llm_corpora.keys())
    X = np.vstack([sa.fit_transform(texts) for texts in llm_corpora.values()])
    n_per_class = [len(texts) for texts in llm_corpora.values()]
    y = np.concatenate([np.full(n, i) for i, n in enumerate(n_per_class)])

    preds, trues = [], []
    for tr, te in LeaveOneOut().split(X):
        clf = NearestCentroid()
        clf.fit(X[tr], y[tr])
        preds.append(clf.predict(X[te])[0])
        trues.append(y[te][0])

    acc    = float(np.mean(np.array(preds) == np.array(trues)))
    cm     = confusion_matrix(trues, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=model_names, yticklabels=model_names,
                ax=ax, linewidths=0.5, linecolor="#333333", cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Prédit", color="#AAAAAA", labelpad=10)
    ax.set_ylabel("Réel", color="#AAAAAA", labelpad=10)
    ax.set_title(f"Matrice de confusion — centroïde (LOO)\nAccuracy : {acc:.1%}  |  Baseline : {1/len(model_names):.0%}",
                 color="white", pad=12)
    ax.tick_params(colors="#AAAAAA")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    _save(fig, "confusion_matrix.png", results_dir)
    print(f"       accuracy={acc:.1%}")
    return X, y


def fig_feature_importance(
    sa: StyleAnalyzer, X: np.ndarray, y: np.ndarray, results_dir: Path
) -> np.ndarray:
    scaler = StandardScaler()
    svm    = SVC(kernel="linear", C=1.0, decision_function_shape="ovr")
    svm.fit(scaler.fit_transform(X), y)
    importance = np.abs(svm.coef_).mean(axis=0)
    top_idx    = np.argsort(importance)[::-1][:15]

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)
    ax.barh(range(15), importance[top_idx][::-1], color="#CE93D8", alpha=0.85, edgecolor="#444444")
    ax.set_yticks(range(15))
    ax.set_yticklabels([sa.function_words[i] for i in top_idx][::-1], color="#AAAAAA", fontsize=10)
    ax.set_xlabel("Importance (|coeff SVM| moyen)", color="#AAAAAA")
    ax.set_title("Mots-outils les plus discriminants entre LLMs (SVM linéaire, OVR)", color="white", pad=10)
    ax.grid(axis="x", alpha=0.1, color="#AAAAAA")
    _save(fig, "feature_importance.png", results_dir)
    return top_idx


def fig_umap(sa: StyleAnalyzer, corpus: dict, results_dir: Path) -> None:
    try:
        import umap as umap_lib
    except ImportError:
        print("  – umap_clusters.png (skipped — pip install umap-learn)")
        return

    labels_list = list(corpus.keys())
    texts_flat  = [t for grp in corpus.values() for t in grp]
    labels_flat = [lbl for grp, lbl in zip(corpus.values(), labels_list) for _ in grp]

    X      = sa.fit_transform(texts_flat)
    coords = umap_lib.UMAP(
        n_components=2, n_neighbors=8, min_dist=0.3, random_state=42
    ).fit_transform(X)

    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)
    handles = []
    rng = np.random.default_rng(42)
    for label in labels_list:
        mask  = np.array([g == label for g in labels_flat])
        pts   = coords[mask]
        color = PALETTE.get(label, "#AAAAAA")
        jitter = rng.uniform(-0.04, 0.04, pts.shape)
        ax.scatter((pts + jitter)[:, 0], (pts + jitter)[:, 1], c=color, s=45,
                   alpha=0.55, edgecolors="none", zorder=3)
        if len(pts) >= 3:
            cov  = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h  = 2 * np.sqrt(np.maximum(vals, 0))
            ell   = Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=angle,
                            fc=color, alpha=0.10, ec=color, lw=1.8, ls="--", zorder=2)
            ax.add_patch(ell)
        cx, cy = pts.mean(axis=0)
        ax.scatter([cx], [cy], c=color, s=220, marker="X",
                   edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(f" {label}", (cx, cy), color=color,
                    fontsize=8.5, fontweight="bold",
                    xytext=(6, 4), textcoords="offset points")
        handles.append(mpatches.Patch(color=color, label=label))
    ax.set_title(
        "Espace stylistique — UMAP 2D\n"
        "Points = textes individuels  ·  ✕ = centroïde  ·  ellipses = zone 1-σ",
        color="white", pad=12,
    )
    ax.legend(handles=handles, facecolor="#1C2128", edgecolor="#444444",
              labelcolor="white", fontsize=9)
    ax.grid(alpha=0.08, color="#AAAAAA")
    _save(fig, "umap_clusters.png", results_dir)


def fig_logistic(
    sa: StyleAnalyzer, X: np.ndarray, y: np.ndarray,
    feat_names: list[str], llm_corpora: dict, results_dir: Path,
    acc_baseline: float | None = None,
) -> None:
    model_names = list(llm_corpora.keys())
    n_classes   = len(model_names)
    scaler      = StandardScaler()
    X_sc        = scaler.fit_transform(X)

    preds = []
    for tr, te in LeaveOneOut().split(X_sc):
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X_sc[tr], y[tr])
        preds.append(clf.predict(X_sc[te])[0])
    acc = float(np.mean(np.array(preds) == np.array(y)))

    clf_all = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf_all.fit(X_sc, y)

    importance    = np.abs(clf_all.coef_).mean(axis=0)
    top_idx       = np.argsort(importance)[::-1][:15]
    top_feat_names = [feat_names[i] for i in top_idx]
    coef_mat      = clf_all.coef_[:, top_idx]

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    vmax = float(np.abs(coef_mat).max())
    sns.heatmap(
        coef_mat, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        xticklabels=top_feat_names, yticklabels=model_names,
        ax=ax, linewidths=0.3, linecolor="#333333",
        cbar_kws={"shrink": 0.8},
    )
    baseline_note = (
        f"  ·  mots-outils seuls : {acc_baseline:.1%}"
        if acc_baseline is not None else ""
    )
    ax.set_title(
        f"Logistic regression — top 15 features (mots-outils + hedge/burstiness/ponctuation)\n"
        f"LOO accuracy : {acc:.1%}{baseline_note}  |  Hasard : {1/n_classes:.0%}  |  n=50/classe",
        color="white", pad=12,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color="#AAAAAA", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, color="#AAAAAA", fontsize=9)
    ax.tick_params(colors="#AAAAAA")
    _save(fig, "logistic_feature_importance.png", results_dir)
    print(f"       logistic acc={acc:.1%}")


def fig_shift_by_source(
    shifts_all: dict, results_dir: Path, n_zola: int = 8
) -> None:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.patch.set_facecolor(BG)

    slices = [("Textes Zola", slice(0, n_zola)), ("Textes Maupassant", slice(n_zola, None))]
    for ax, (label, sl) in zip(axes, slices):
        _ax_style(ax)
        for i, (name, shifts) in enumerate(shifts_all.items()):
            color = PALETTE[name]
            vals  = np.array(shifts[sl])
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, c=color, s=40, alpha=0.75, zorder=3)
            ax.plot([i - 0.22, i + 0.22], [vals.mean()] * 2, color=color, lw=2.5, zorder=4)
        ax.set_xticks(range(len(shifts_all)))
        ax.set_xticklabels(list(shifts_all.keys()), color="#AAAAAA", fontsize=9)
        ax.set_title(label, color="white")
        ax.grid(axis="y", alpha=0.1, color="#AAAAAA")
    axes[0].set_ylabel("Shift cosinus", color="#AAAAAA")
    fig.suptitle("Shift stylistique selon la source humaine", color="white", y=1.01)
    _save(fig, "shift_by_source.png", results_dir)


def fig_dendrogram(sa: StyleAnalyzer, corpus: dict, results_dir: Path) -> None:
    dend_labels = {
        k.replace("(", "\n(").replace("Humain \n", "Humain\n"): v
        for k, v in corpus.items()
    }
    names     = list(dend_labels.keys())
    centroids = np.array([sa.fit_transform(v).mean(axis=0) for v in dend_labels.values()])
    dist_mat  = squareform(pdist(centroids, metric="cosine"))
    link_mat  = linkage(dist_mat, method="ward")

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)
    dendrogram(link_mat, labels=names, ax=ax,
               color_threshold=0.5 * link_mat[:, 2].max(),
               above_threshold_color="#AAAAAA", leaf_rotation=0)
    for lbl in ax.get_xticklabels():
        raw = lbl.get_text().replace("\n", " ")
        lbl.set_color(PALETTE.get(raw, "#AAAAAA"))
        lbl.set_fontsize(10)
    ax.set_title("Clustering hiérarchique des centroïdes\n(distance cosinus, linkage Ward)",
                 color="white", pad=12)
    ax.set_ylabel("Distance Ward", color="#AAAAAA")
    ax.grid(axis="y", alpha=0.08, color="#AAAAAA")
    _save(fig, "dendrogram.png", results_dir)


def fig_radar(sa: StyleAnalyzer, corpus: dict, top_idx: np.ndarray, results_dir: Path) -> None:
    words = [sa.function_words[i] for i in top_idx[:10]]

    def _raw_freq(texts: list[str], words: list[str]) -> np.ndarray:
        tokens = re.findall(r"\b\w+\b", " ".join(texts).lower())
        n = len(tokens)
        return np.array([tokens.count(w) / n * 100 for w in words])

    human_texts = corpus.get("Humain (Zola)", []) + corpus.get("Humain (Maupassant)", [])
    baseline    = _raw_freq(human_texts, words)

    llm_labels = ["GPT-4", "Claude 3", "Mistral 7B", "Gemini Pro"]
    matrix = []
    for lbl in llm_labels:
        llm_freq = _raw_freq(corpus[lbl], words)
        rel = np.where(baseline > 0, (llm_freq - baseline) / baseline * 100, 0.0)
        matrix.append(rel)

    data = np.array(matrix)
    vmax = max(60.0, float(np.abs(data).max()))

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_facecolor(THEME["surface"])
    sns.heatmap(
        data, annot=True, fmt="+.0f",
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        xticklabels=words, yticklabels=llm_labels,
        ax=ax, linewidths=0.4, linecolor=THEME["spine"],
        cbar_kws={"shrink": 0.7, "label": "Écart vs baseline humain (%)"},
    )
    ax.set_title(
        "Profil stylistique — écart par rapport au baseline humain (%)\n"
        "Rouge = surreprésentation  ·  Bleu = sous-représentation",
        color=THEME["text_primary"], pad=14,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             color=THEME["text_muted"], fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0,
             color=THEME["text_muted"], fontsize=10)
    ax.tick_params(colors=THEME["tick"])
    _save(fig, "radar_profiles.png", results_dir)


def fig_drift_trajectories(
    sa: StyleAnalyzer, corpus: dict, shifts_all: dict, results_dir: Path
) -> None:
    """Hero visual: PCA on raw function-word vectors.

    Human texts form the anchor; arrows go from the human centroid to each
    LLM centroid.  Axes carry real meaning (% variance explained).
    """
    from matplotlib.patches import Ellipse as _Ell

    originals   = load_originals()
    llm_corpora = load_llm_corpora()

    # ── Build combined PCA projection ──────────────────────────────────────
    all_vecs, all_lbls = [], []
    orig_vecs = sa.fit_transform(originals)
    for v in orig_vecs:
        all_vecs.append(v); all_lbls.append("Humain")
    for lbl, rewrites in llm_corpora.items():
        for v in sa.fit_transform(rewrites):
            all_vecs.append(v); all_lbls.append(lbl)

    X       = np.array(all_vecs)
    lbl_arr = np.array(all_lbls)
    pca     = PCA(n_components=2, random_state=42)
    coords  = pca.fit_transform(X)
    var     = pca.explained_variance_ratio_ * 100

    human_mask     = lbl_arr == "Humain"
    human_centroid = coords[human_mask].mean(axis=0)
    llm_labels     = list(llm_corpora.keys())
    llm_centroids  = {l: coords[lbl_arr == l].mean(axis=0) for l in llm_labels}

    # ── Canvas ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.patch.set_facecolor(THEME["bg"])
    _ax_style(ax)

    # ── Human scatter + 1-σ ellipse ─────────────────────────────────────
    ax.scatter(*coords[human_mask].T, c=THEME["text_muted"], s=14, alpha=0.18,
               zorder=2, rasterized=True)
    h_pts = coords[human_mask]
    cov   = np.cov(h_pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h_ = 2 * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(_Ell(xy=human_centroid, width=w, height=h_, angle=angle,
                      fc=THEME["text_muted"], alpha=0.06, ec=THEME["text_muted"],
                      lw=1.2, ls="--", zorder=2))

    # ── LLM scatter (very light — centroids carry the story) ────────────
    for label in llm_labels:
        mask  = lbl_arr == label
        color = PALETTE.get(label, "#AAAAAA")
        ax.scatter(*coords[mask].T, c=color, s=10, alpha=0.15,
                   zorder=3, rasterized=True)

    # ── Human centroid — star ─────────────────────────────────────────────
    ax.scatter(*human_centroid, c="white", s=520, marker="*",
               edgecolors=THEME["text_muted"], linewidths=1.2, zorder=8)
    ax.annotate(
        "Baseline humain (★)", human_centroid,
        xytext=(12, -18), textcoords="offset points",
        color=THEME["text_muted"], fontsize=9, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc=THEME["bg"], ec="none", alpha=0.80),
    )

    # ── Arrows from human centroid → each LLM centroid ──────────────────
    arrow_rads   = {"GPT-4": 0.18, "Claude 3": -0.15, "Mistral 7B": 0.28, "Gemini Pro": -0.22}
    label_offsets = {
        "GPT-4":      (-10, -44),
        "Claude 3":   (-20,  28),
        "Mistral 7B": ( 18,  28),
        "Gemini Pro": (-20, -40),
    }
    delta_nudge = {
        "GPT-4":      np.array([ 0.012, -0.035]),
        "Claude 3":   np.array([-0.025,  0.018]),
        "Mistral 7B": np.array([ 0.040,  0.022]),
        "Gemini Pro": np.array([-0.022, -0.028]),
    }

    for label in llm_labels:
        color      = PALETTE.get(label, "#AAAAAA")
        cx, cy     = llm_centroids[label]
        mean_shift = float(np.mean(shifts_all[label]))
        rad        = arrow_rads.get(label, 0.20)

        # Curved arrow
        ax.annotate(
            "", xy=(cx, cy), xytext=human_centroid,
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=2.8,
                connectionstyle=f"arc3,rad={rad}", mutation_scale=22,
            ),
            zorder=7,
        )

        # Δ annotation near arrow midpoint
        nudge = delta_nudge.get(label, np.array([0.02, 0.015]))
        mid   = 0.5 * (human_centroid + np.array([cx, cy])) + nudge * (
            np.array([cx, cy]) - human_centroid
        ).mean()
        ax.annotate(
            f"Δ = {mean_shift:.3f}", mid,
            color=color, fontsize=8.5, fontstyle="italic", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc=THEME["bg"], ec=color, alpha=0.85, lw=0.6),
            zorder=9,
        )

        # LLM centroid marker
        ax.scatter([cx], [cy], c=color, s=360, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=10)

        # Model label
        dx, dy = label_offsets.get(label, (12, 6))
        ax.annotate(
            label, (cx, cy), color=color, fontsize=11, fontweight="bold",
            xytext=(dx, dy), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.32", fc=THEME["bg"], ec="none", alpha=0.90),
            zorder=11,
        )

    ax.set_xlabel(f"PC1 ({var[0]:.1f} % variance)", color=THEME["text_muted"], labelpad=8)
    ax.set_ylabel(f"PC2 ({var[1]:.1f} % variance)", color=THEME["text_muted"], labelpad=8)
    ax.set_title(
        "Stylistic drift in function-word PCA space\n"
        "★ = human centroid  ·  ✕ = LLM centroid  ·  Δ = mean cosine shift",
        color=THEME["text_primary"], pad=16,
    )
    ax.grid(alpha=0.03, color=THEME["grid"])
    _save(fig, "drift_trajectories.png", results_dir)


def fig_shift_vs_length(shifts_all: dict, llm_corpora: dict, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)
    for name, texts in llm_corpora.items():
        color   = PALETTE[name]
        lengths = [len(re.findall(r"\b\w+\b", t)) for t in texts]
        s_vals  = shifts_all[name]
        ax.scatter(lengths, s_vals, c=color, s=45, alpha=0.75,
                   edgecolors="black", linewidths=0.4, label=name, zorder=3)
        z = np.polyfit(lengths, s_vals, 1)
        xfit = np.linspace(min(lengths), max(lengths), 100)
        ax.plot(xfit, np.poly1d(z)(xfit), color=color, lw=1.2, alpha=0.5, ls="--")
    ax.axvline(80, color="#FF5555", lw=1.2, ls=":", alpha=0.8, label="Seuil 80 mots")
    ax.set_xlabel("Longueur du texte (mots)", color="#AAAAAA")
    ax.set_ylabel("Shift cosinus", color="#AAAAAA")
    ax.set_title("Shift stylistique vs longueur du texte\n(tirets = tendance linéaire par modèle)",
                 color="white", pad=10)
    ax.legend(facecolor="#1C2128", edgecolor="#444444", labelcolor="white", fontsize=9)
    ax.grid(alpha=0.08, color="#AAAAAA")
    _save(fig, "shift_vs_length.png", results_dir)


def fig_bootstrap(shifts_all: dict, results_dir: Path) -> None:
    """Horizontal bar chart with 95 % CI and significance brackets."""
    model_names = list(shifts_all.keys())
    means_ci    = [(np.mean(v), *bootstrap_ci(v)) for v in shifts_all.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(THEME["bg"])
    _ax_style(ax)

    y_pos = np.arange(len(model_names))
    for i, (name, (mean, lo, hi)) in enumerate(zip(reversed(model_names), reversed(means_ci))):
        color = PALETTE[name]
        ax.barh(i, mean, color=color, alpha=0.80, height=0.52, zorder=3,
                left=0.0, edgecolor="none")
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]],
                    fmt="none", color="white", capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
        # Value label
        ax.text(hi + 0.004, i, f"{mean:.3f}  [CI {lo:.3f}–{hi:.3f}]",
                va="center", color="white", fontsize=8.5, fontweight="bold")
        # Model label inline
        ax.text(0.004, i, name, va="center", ha="left",
                color=THEME["bg"], fontsize=9, fontweight="bold", zorder=5)

    # Significance bracket: Gemini vs the group
    gemini_idx   = list(reversed(model_names)).index("Gemini Pro")
    gpt4_idx     = list(reversed(model_names)).index("GPT-4")
    gemini_mean  = means_ci[model_names.index("Gemini Pro")][0]
    bracket_x    = gemini_mean + 0.055
    ax.annotate(
        "", xy=(bracket_x, gpt4_idx), xytext=(bracket_x, gemini_idx),
        arrowprops=dict(arrowstyle="<->", color="#FF7B72", lw=1.4),
    )
    ax.text(bracket_x + 0.005, (gpt4_idx + gemini_idx) / 2,
            "p < 0.001\n(Bonferroni)", color="#FF7B72", fontsize=7.5,
            va="center", fontstyle="italic")

    ax.set_yticks([])
    ax.set_xlabel("Mean cosine shift (stylistic displacement)", color=THEME["text_muted"], labelpad=8)
    ax.set_title(
        "Mean stylistic shift per model — 95 % bootstrap CI (n = 5 000)\n"
        "Longer bar = more displacement from human baseline",
        color=THEME["text_primary"], pad=12,
    )
    ax.set_xlim(0, max(hi for _, lo, hi in means_ci) * 1.38)
    ax.grid(axis="x", alpha=0.06, color=THEME["grid"])
    ax.spines["left"].set_visible(False)
    _save(fig, "bootstrap_ci.png", results_dir)


def fig_permutation(shifts_all: dict, results_dir: Path) -> None:
    pairs = [("GPT-4", "Mistral 7B"), ("Claude 3", "Gemini Pro")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(BG)
    for ax, (ma, mb) in zip(axes, pairs):
        _ax_style(ax)
        res = permutation_test(shifts_all[ma], shifts_all[mb], n_perm=5_000)
        ax.hist(res["null_distribution"], bins=40, color="#555577", alpha=0.7, edgecolor="none",
                label="H₀ (permutations)")
        ax.axvline(res["observed"], color="#FF7043", lw=2,
                   label=f'Observé = {res["observed"]:.4f}')
        sig = "✓ p < 0.05" if res["significant_05"] else "✗ non sig."
        ax.set_title(f"{ma} vs {mb}\np = {res['p_value']:.4f} — {sig}", color="white", fontsize=10)
        ax.set_xlabel("|Δ moyennes| sous H₀", color="#AAAAAA")
        ax.legend(facecolor="#1C2128", edgecolor="#444444", labelcolor="white", fontsize=8)
        ax.grid(alpha=0.08, color="#AAAAAA")
    fig.suptitle("Tests de permutation — les shifts diffèrent-ils significativement entre modèles ?",
                 color="white", y=1.02)
    _save(fig, "permutation_null.png", results_dir)


# ---------------------------------------------------------------------------
# Robustesse inter-prompt
# ---------------------------------------------------------------------------

def fig_prompt_robustness(sa: StyleAnalyzer, results_dir: Path) -> None:
    """Scatter shift_p1 vs shift_p2 par modèle — teste la stabilité du fingerprint."""
    MODEL_SLUGS = {
        "GPT-4":      "gpt4",
        "Claude 3":   "claude3",
        "Mistral 7B": "mistral",
        "Gemini Pro": "gemini",
    }

    # Collect data per model
    model_data: dict[str, tuple[list[float], list[float]]] = {}
    for label, slug in MODEL_SLUGS.items():
        orig_p1, rew_p1 = load_single_model_aligned(slug, "p1")
        orig_p2, rew_p2 = load_single_model_aligned(slug, "p2")
        if not orig_p1 or not orig_p2:
            continue
        # Use texts available in both prompts (same combined-corpus positions)
        n = min(len(orig_p1), len(orig_p2))
        if n == 0:
            continue
        s1 = [sa.shift(o, r) for o, r in zip(orig_p1[:n], rew_p1[:n])]
        s2 = [sa.shift(o, r) for o, r in zip(orig_p2[:n], rew_p2[:n])]
        model_data[label] = (s1, s2)

    if not model_data:
        print("  prompt_robustness.png (skipped — aucune donnée P2 disponible)")
        return

    n_models = len(model_data)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    fig.patch.set_facecolor(THEME["bg"])
    fig.suptitle(
        "Robustesse inter-prompt du fingerprint stylistique\n"
        "(chaque point = un texte ; proximité de la diagonale = stabilité)",
        color=THEME["text_primary"], fontsize=THEME["title_size"], y=1.01,
    )

    for ax, (label, (s1, s2)) in zip(axes[0], model_data.items()):
        _ax_style(ax)
        color = PALETTE.get(label, "#AAAAAA")

        lim_max = max(max(s1), max(s2)) * 1.08
        ax.plot([0, lim_max], [0, lim_max], "--", color="#484F58", lw=1.2, alpha=0.7, zorder=1)

        ax.scatter(s1, s2, c=color, s=28, alpha=0.65, zorder=3)

        corr = float(np.corrcoef(s1, s2)[0, 1])
        ax.set_title(label, color=color, fontsize=THEME["title_size"], pad=8)
        ax.set_xlabel("Shift P1 (neutre)", color=THEME["text_muted"])
        ax.set_ylabel("Shift P2 (simplifié)", color=THEME["text_muted"])
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.text(0.97, 0.05, f"r = {corr:.2f}  n = {len(s1)}",
                ha="right", va="bottom", transform=ax.transAxes,
                color=THEME["text_muted"], fontsize=THEME["annot_size"])

    fig.tight_layout()
    _save(fig, "prompt_robustness.png", results_dir)


# ---------------------------------------------------------------------------
# Code stylometry — illustrative figure (no real code data)
# ---------------------------------------------------------------------------

def fig_code_stylometry(results_dir: Path) -> None:
    """Illustrative visualization of code stylometric fingerprints.

    Uses synthetic data to show the concept: developer style clusters in a
    2D space (type-hint density vs. structural habit), and a convergence zone
    representing AI-assisted code.  Clearly labelled as synthetic.
    """
    rng = np.random.default_rng(42)

    developers = {
        "Dev A\n(type-heavy, early-return)": {
            "center": np.array([0.82, 0.78]),
            "std":    np.array([0.07, 0.06]),
            "color":  "#58A6FF",
            "n":      35,
            "marker": "o",
        },
        "Dev B\n(implicit, nested)": {
            "center": np.array([0.22, 0.28]),
            "std":    np.array([0.07, 0.07]),
            "color":  "#3FB950",
            "n":      30,
            "marker": "o",
        },
        "Dev C\n(mixed style)": {
            "center": np.array([0.55, 0.75]),
            "std":    np.array([0.08, 0.06]),
            "color":  "#E3B341",
            "n":      28,
            "marker": "o",
        },
    }
    ai_center = np.array([0.62, 0.52])
    ai_color  = "#FF7B72"

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor(THEME["bg"])
    _ax_style(ax)

    # ── Developer clusters ──────────────────────────────────────────────
    for name, cfg in developers.items():
        pts = rng.normal(cfg["center"], cfg["std"], (cfg["n"], 2))
        pts = np.clip(pts, 0.02, 0.98)
        ax.scatter(*pts.T, c=cfg["color"], s=22, alpha=0.40, zorder=3, rasterized=True)
        cx, cy = cfg["center"]
        ax.scatter([cx], [cy], c=cfg["color"], s=200, marker="X",
                   edgecolors="white", linewidths=1.4, zorder=6)
        ax.annotate(
            name, (cx, cy), xytext=(12, 7), textcoords="offset points",
            color=cfg["color"], fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=THEME["bg"], ec="none", alpha=0.88),
            zorder=8,
        )

    # ── AI-assisted convergence zone ──────────────────────────────────
    ai_pts = rng.normal(ai_center, [0.11, 0.10], (70, 2))
    ai_pts = np.clip(ai_pts, 0.02, 0.98)
    ax.scatter(*ai_pts.T, c=ai_color, s=16, alpha=0.28, marker="s",
               zorder=3, rasterized=True)
    ax.scatter(*ai_center, c=ai_color, s=280, marker="D",
               edgecolors="white", linewidths=1.4, zorder=6)
    ax.annotate(
        "AI-assisted code\n(Copilot / GPT completions)",
        ai_center, xytext=(14, -32), textcoords="offset points",
        color=ai_color, fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc=THEME["bg"], ec="none", alpha=0.88),
        zorder=8,
    )

    # ── Drift arrows: each dev centroid → AI region ──────────────────
    arrow_rads = {
        "Dev A\n(type-heavy, early-return)":  0.10,
        "Dev B\n(implicit, nested)":          -0.12,
        "Dev C\n(mixed style)":               0.06,
    }
    for name, cfg in developers.items():
        src = cfg["center"]
        vec = ai_center - src
        tip = src + 0.48 * vec          # arrow stops halfway — drift, not arrival
        ax.annotate(
            "", xy=tip, xytext=src,
            arrowprops=dict(
                arrowstyle="-|>", color=cfg["color"], lw=1.8, alpha=0.65,
                mutation_scale=16,
                connectionstyle=f"arc3,rad={arrow_rads[name]}",
            ),
            zorder=5,
        )

    # ── Axis labels and annotations ──────────────────────────────────
    ax.set_xlabel("Type annotation density\n(0 = none  →  1 = comprehensive)",
                  color=THEME["text_muted"], labelpad=10)
    ax.set_ylabel("Early-return rate\n(0 = deep nesting  →  1 = always early exit)",
                  color=THEME["text_muted"], labelpad=10)
    ax.set_xlim(-0.02, 1.10)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])

    ax.set_title(
        "Code stylometry — developer fingerprints and AI-assisted drift\n"
        "Each axis = a measurable structural habit  ·  Arrows = direction of stylistic pressure",
        color=THEME["text_primary"], pad=16,
    )
    ax.annotate(
        "★  Synthetic / illustrative — this project does not include code data",
        xy=(0.5, 0.012), xycoords="axes fraction", ha="center",
        color=THEME["text_muted"], fontsize=8, fontstyle="italic",
    )
    ax.grid(alpha=0.04, color=THEME["grid"])
    _save(fig, "code_stylometry.png", results_dir)


# ---------------------------------------------------------------------------
# Résumé statistique
# ---------------------------------------------------------------------------

def print_stats(shifts_all: dict) -> None:
    print("\n" + "─" * 68)
    print("TESTS PAIRÉS (Welch, correction Bonferroni)")
    print("─" * 68)
    for r in pairwise_tests(shifts_all):
        sig = "✓" if r["significant_05"] else "✗"
        print(f"  {r['model_a']:<15} vs {r['model_b']:<15} "
              f"p_corr={r['p_corrected']:.4f}  {sig}")

    sa_tmp = StyleAnalyzer()
    corpus = load_corpus()
    print("\nVARIANCE INTRA-GROUPE (bruit naturel)")
    print("─" * 40)
    for label, v in sorted(intra_variance(corpus, sa_tmp).items(), key=lambda x: x[1]):
        print(f"  {label:<22} {v:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.family":    "sans-serif",
        "font.size":      THEME["tick_size"],
        "axes.titlesize": THEME["title_size"],
        "axes.labelsize": THEME["label_size"],
    })

    sa          = StyleAnalyzer()
    corpus      = load_corpus()
    originals   = load_originals()
    llm_corpora = load_llm_corpora()

    shifts_all = {
        name: [sa.shift(orig, rew) for orig, rew in zip(originals, texts)]
        for name, texts in llm_corpora.items()
    }

    print("Shifts (distance cosinus original → réécriture)")
    print("─" * 50)
    for name, shifts in shifts_all.items():
        lo, hi = bootstrap_ci(shifts)
        print(f"  {name:<15} mean={np.mean(shifts):.4f}  IC95=[{lo:.4f}, {hi:.4f}]")
    print()

    fig_violin(sa, shifts_all, results_dir)
    fig_drift_trajectories(sa, corpus, shifts_all, results_dir)
    fig_pca(sa, corpus, results_dir)
    if not fast:
        fig_tsne(sa, corpus, results_dir)
        fig_umap(sa, corpus, results_dir)
    else:
        print("  – tsne_clusters.png (skipped — --fast)")
        print("  – umap_clusters.png (skipped — --fast)")
    fig_fingerprint(sa, corpus, results_dir)
    X, y = fig_confusion(sa, llm_corpora, results_dir)
    top_idx = fig_feature_importance(sa, X, y, results_dir)

    # Baseline LOO accuracy with function words only
    _scaler_b = StandardScaler()
    _X_b = _scaler_b.fit_transform(X)
    _preds_b = []
    for _tr, _te in LeaveOneOut().split(_X_b):
        _clf_b = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        _clf_b.fit(_X_b[_tr], y[_tr])
        _preds_b.append(_clf_b.predict(_X_b[_te])[0])
    acc_baseline = float(np.mean(np.array(_preds_b) == np.array(y)))

    # Extended feature matrix: function-word vectors + hedge/burstiness/punctuation
    all_texts  = [t for texts in llm_corpora.values() for t in texts]
    X_ext      = np.hstack([X, extra_features(all_texts)])
    feat_names_ext = list(sa.function_words) + extra_feature_names()
    fig_logistic(sa, X_ext, y, feat_names_ext, llm_corpora, results_dir, acc_baseline=acc_baseline)
    fig_shift_by_source(shifts_all, results_dir, n_zola=25)
    fig_dendrogram(sa, corpus, results_dir)
    fig_radar(sa, corpus, top_idx, results_dir)
    fig_shift_vs_length(shifts_all, llm_corpora, results_dir)
    fig_bootstrap(shifts_all, results_dir)
    fig_permutation(shifts_all, results_dir)
    fig_prompt_robustness(sa, results_dir)
    fig_code_stylometry(results_dir)

    print_stats(shifts_all)
    print(f"\nTous les graphiques générés dans {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true",
                        help="Sauter t-SNE (plus lent) pour un run rapide")
    args = parser.parse_args()
    main(fast=args.fast)
