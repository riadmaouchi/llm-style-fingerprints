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
from src.data import load_corpus, load_originals, load_llm_corpora
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
    from matplotlib.patches import Ellipse
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # Rebuild LDA shift-vector space (same as fig_pca)
    originals   = load_originals()
    llm_corpora = load_llm_corpora()
    orig_vecs   = sa.fit_transform(originals)

    shift_pts, shift_lbls = [], []
    for lbl, rewrites in llm_corpora.items():
        rew_vecs = sa.fit_transform(rewrites)
        for ov, rv in zip(orig_vecs, rew_vecs):
            shift_pts.append(rv - ov)
            shift_lbls.append(lbl)

    S       = np.array(shift_pts)
    lbl_arr = np.array(shift_lbls)
    lda     = LinearDiscriminantAnalysis(n_components=2)
    coords  = lda.fit_transform(S, lbl_arr)

    llm_labels    = list(llm_corpora.keys())
    llm_centroids = {l: coords[lbl_arr == l].mean(axis=0) for l in llm_labels}
    origin        = np.array([0.0, 0.0])  # neutral reference in LDA shift space

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(THEME["bg"])
    _ax_style(ax)

    # 1-sigma ellipses per LLM (show spread without scatter noise)
    from matplotlib.patches import Ellipse as _Ell
    for label in llm_labels:
        mask  = lbl_arr == label
        pts   = coords[mask]
        color = PALETTE.get(label, "#AAAAAA")
        if len(pts) >= 3:
            cov  = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h  = 2 * np.sqrt(np.maximum(vals, 0))
            ax.add_patch(_Ell(xy=llm_centroids[label], width=w, height=h, angle=angle,
                              fc=color, alpha=0.08, ec=color, lw=1.5, ls="--", zorder=2))

    # Origin — geometric reference of LDA shift space
    ax.scatter(*origin, c=THEME["text_muted"], s=180, marker="o",
               zorder=6, edgecolors="white", linewidths=1.0, alpha=0.55)
    ax.annotate("réf.", origin, xytext=(6, 8), textcoords="offset points",
                color=THEME["text_muted"], fontsize=8, fontstyle="italic")

    # Arrows from origin → each LLM centroid + Δ labels
    rads    = {"GPT-4": 0.30, "Claude 3": -0.20, "Mistral 7B": 0.55, "Gemini Pro": -0.45}
    lbl_off = {
        "GPT-4":      (14,  -20),
        "Claude 3":   (-14,  16),
        "Mistral 7B": ( 14,  14),
        "Gemini Pro": (-14, -20),
    }
    delta_off = {
        "GPT-4":      np.array([ 0.28, -0.22]),
        "Claude 3":   np.array([-0.30,  0.20]),
        "Mistral 7B": np.array([ 0.30,  0.18]),
        "Gemini Pro": np.array([-0.26, -0.26]),
    }

    for label in llm_labels:
        color      = PALETTE.get(label, "#AAAAAA")
        cx, cy     = llm_centroids[label]
        mean_shift = float(np.mean(shifts_all[label]))
        rad        = rads.get(label, 0.25)

        ax.annotate(
            "", xy=(cx, cy), xytext=origin,
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=2.6,
                connectionstyle=f"arc3,rad={rad}", mutation_scale=20,
            ),
            zorder=7,
        )

        doff = delta_off.get(label, np.array([0.20, 0.15]))
        mid  = origin + 0.5 * (np.array([cx, cy]) - origin) + doff
        ax.annotate(
            f"Δ = {mean_shift:.2f}", mid,
            color=color, fontsize=9, fontstyle="italic", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=THEME["bg"], ec="none", alpha=0.82),
            zorder=9,
        )

        ax.scatter([cx], [cy], c=color, s=340, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=10)
        dx, dy = lbl_off.get(label, (10, 5))
        ax.annotate(label, (cx, cy), color=color, fontsize=10.5, fontweight="bold",
                    xytext=(dx, dy), textcoords="offset points")

    # Zoom on the centroid zone, clip outlier points
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("LD1 — axe de discrimination inter-LLM", color=THEME["text_muted"], labelpad=8)
    ax.set_ylabel("LD2 — axe de discrimination inter-LLM", color=THEME["text_muted"], labelpad=8)
    ax.set_title(
        "Trajectoires de drift stylistique — espace LDA (vecteurs de shift)\n"
        "Flèches : référence → centroïde LLM  ·  Δ = shift cosinus moyen  ·  ellipses = dispersion 1-σ",
        color=THEME["text_primary"], pad=14,
    )
    ax.grid(alpha=0.04, color=THEME["grid"])
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
    model_names = list(shifts_all.keys())
    means_ci    = [(np.mean(v), *bootstrap_ci(v)) for v in shifts_all.values()]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)
    for i, (name, (mean, lo, hi)) in enumerate(zip(model_names, means_ci)):
        color = PALETTE[name]
        ax.bar(i, mean, color=color, alpha=0.75, width=0.5, zorder=3)
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]],
                    fmt="none", color="white", capsize=6, capthick=1.5, elinewidth=1.5, zorder=4)
        ax.text(i, hi + 0.005, f"{mean:.3f}", ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, color="#AAAAAA")
    ax.set_ylabel("Shift cosinus moyen", color="#AAAAAA")
    ax.set_title("Shift moyen + IC 95 % Bootstrap (n=5 000)", color="white", pad=10)
    ax.grid(axis="y", alpha=0.1, color="#AAAAAA")
    ax.set_ylim(0, max(hi for _, lo, hi in means_ci) * 1.2)
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

    print_stats(shifts_all)
    print(f"\nTous les graphiques générés dans {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true",
                        help="Sauter t-SNE (plus lent) pour un run rapide")
    args = parser.parse_args()
    main(fast=args.fast)
