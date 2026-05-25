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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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

BG = "#0D1117"
plt.style.use("dark_background")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str, results_dir: Path) -> None:
    path = results_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {name}")


def _ax_style(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors="#555555")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_violin(sa: StyleAnalyzer, shifts_all: dict, results_dir: Path) -> None:
    fig = sa.plot_shift_violin(
        shifts_all,
        title="Distribution des shifts stylistiques par LLM\n(distance cosinus : original → réécriture)",
    )
    _save(fig, "shift_distributions.png", results_dir)


def fig_pca(sa: StyleAnalyzer, corpus: dict, results_dir: Path) -> None:
    fig = sa.plot_clusters(
        texts_groups=list(corpus.values()),
        labels=list(corpus.keys()),
        title="Espace stylistique — projection PCA 2D\nLes LLMs forment des clusters distincts des humains",
    )
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

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)
    handles = []
    for label in labels_list:
        mask = np.array([g == label for g in labels_flat])
        pts   = coords[mask]
        color = PALETTE.get(label, "#AAAAAA")
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=90, alpha=0.82,
                   edgecolors="black", linewidths=0.5, zorder=3)
        cx, cy = pts.mean(axis=0)
        ax.scatter([cx], [cy], c=color, s=320, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(f"  {label}", (cx, cy), color=color, fontsize=9, fontweight="bold")
        handles.append(mpatches.Patch(color=color, label=label))
    ax.set_title("Espace stylistique — t-SNE 2D\n(non-linéaire — meilleure séparation des clusters)",
                 color="white", pad=12)
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

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)
    handles = []
    for label in labels_list:
        mask  = np.array([g == label for g in labels_flat])
        pts   = coords[mask]
        color = PALETTE.get(label, "#AAAAAA")
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=90, alpha=0.82,
                   edgecolors="black", linewidths=0.5, zorder=3)
        cx, cy = pts.mean(axis=0)
        ax.scatter([cx], [cy], c=color, s=320, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(f"  {label}", (cx, cy), color=color, fontsize=9, fontweight="bold")
        handles.append(mpatches.Patch(color=color, label=label))
    ax.set_title(
        "Espace stylistique — UMAP 2D\n(meilleure préservation de la structure locale vs PCA/t-SNE)",
        color="white", pad=12,
    )
    ax.legend(handles=handles, facecolor="#1C2128", edgecolor="#444444",
              labelcolor="white", fontsize=9)
    ax.grid(alpha=0.08, color="#AAAAAA")
    _save(fig, "umap_clusters.png", results_dir)


def fig_logistic(
    sa: StyleAnalyzer, X: np.ndarray, y: np.ndarray,
    llm_corpora: dict, results_dir: Path,
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

    importance = np.abs(clf_all.coef_).mean(axis=0)
    top_idx    = np.argsort(importance)[::-1][:15]
    feat_names = [sa.function_words[i] for i in top_idx]
    coef_mat   = clf_all.coef_[:, top_idx]

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    vmax = float(np.abs(coef_mat).max())
    sns.heatmap(
        coef_mat, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        xticklabels=feat_names, yticklabels=model_names,
        ax=ax, linewidths=0.3, linecolor="#333333",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        f"Logistic regression — coefficients par classe, top 15 mots-outils\n"
        f"LOO accuracy : {acc:.1%}  |  Baseline : {1/n_classes:.0%}"
        f"  |  ⚠ n=16/classe — coefficients indicatifs",
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
    radar_words = [sa.function_words[i] for i in top_idx[:8]]

    def _raw_freq(texts: list[str], words: list[str]) -> list[float]:
        tokens = re.findall(r"\b\w+\b", " ".join(texts).lower())
        n = len(tokens)
        return [tokens.count(w) / n * 100 for w in words]

    N      = len(radar_words)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#111820")
    for label, texts in corpus.items():
        vals   = _raw_freq(texts, radar_words) + [_raw_freq(texts, radar_words)[0]]
        color  = PALETTE.get(label, "#AAAAAA")
        ax.plot(angles, vals, "o-", linewidth=2, color=color, markersize=4, label=label)
        ax.fill(angles, vals, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_words, color="#DDDDDD", fontsize=11)
    ax.set_yticklabels([])
    ax.set_title("Profil stylistique par groupe\n(fréquences % des mots-outils discriminants)",
                 color="white", pad=20)
    ax.grid(color="#333333", alpha=0.5)
    ax.spines["polar"].set_color("#333333")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#1C2128", edgecolor="#444444", labelcolor="white", fontsize=9)
    _save(fig, "radar_profiles.png", results_dir)


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
    fig_logistic(sa, X, y, llm_corpora, results_dir)
    fig_shift_by_source(shifts_all, results_dir)
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
