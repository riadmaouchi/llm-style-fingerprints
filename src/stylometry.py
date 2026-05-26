"""
src/stylometry.py
~~~~~~~~~~~~~~~~~

Wrapper autour de stylometry-python (le moteur officiel de cette étude).
Ajoute uniquement ce qui n'est pas dans la lib :
  - PALETTE (couleurs par groupe)
  - fit_transform()  alias de vectorize_batch()
  - plot_shift_violin()

Prérequis :
    pip install stylometry-python

Utilisation :
    from src.stylometry import StyleAnalyzer, PALETTE, FUNCTION_WORDS_FR
"""

import matplotlib.pyplot as plt
import numpy as np

from stylometry import FUNCTION_WORDS_FR
from stylometry import StyleAnalyzer as _Base

# ─── Palette cohérente pour tous les notebooks ───────────────────────────────

PALETTE = {
    "Humain (Zola)":        "#58A6FF",  # bleu GitHub
    "Humain (Maupassant)":  "#3FB950",  # vert GitHub
    "GPT-4":                "#FF7B72",  # rouge-orangé
    "Claude 3":             "#D2A8FF",  # violet clair
    "Mistral 7B":           "#E3B341",  # ambre
    "Gemini Pro":           "#39D353",  # vert distinct
}


# ─── Sous-classe étendue ─────────────────────────────────────────────────────

class StyleAnalyzer(_Base):
    """
    stylometry.StyleAnalyzer + extensions pour cette étude.

    Paramètre supplémentaire :
        min_words=20  (défaut lib : 50 — trop strict pour nos extraits)
    """

    def __init__(self, function_words=None, language="fr", min_words=20):
        super().__init__(function_words=function_words,
                         language=language, min_words=min_words)

    # alias pour compatibilité avec les notebooks
    def fit_transform(self, texts: list) -> np.ndarray:
        """Alias de vectorize_batch()."""
        return self.vectorize_batch(texts)

    # ──────────────────────────────────────────────────────────────────────────
    def plot_shift_violin(
        self,
        shifts_dict: dict,
        title: str = "Distribution des shifts stylistiques par LLM",
        figsize=(10, 5),
    ):
        """
        Violin plot des distributions de shifts cosinus.

        Paramètres
        ----------
        shifts_dict : {"GPT-4": [0.21, 0.19, ...], "Claude 3": [...], ...}
        title       : titre du graphique
        figsize     : taille de la figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")

        labels = list(shifts_dict.keys())
        data   = [shifts_dict[k] for k in labels]
        colors = [PALETTE.get(k, "#AAAAAA") for k in labels]

        parts = ax.violinplot(data, positions=range(len(labels)), widths=0.65,
                              showmeans=True, showmedians=False, showextrema=True)

        for pc, color in zip(parts["bodies"], colors, strict=False):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        for part in ("cmeans", "cmins", "cmaxes", "cbars"):
            if part in parts:
                parts[part].set_color("#FFFFFF")
                parts[part].set_linewidth(1.2)

        # points individuels avec jitter
        rng = np.random.default_rng(42)
        for i, (vals, color) in enumerate(zip(data, colors, strict=False)):
            jitter = rng.uniform(-0.06, 0.06, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       c=color, s=22, alpha=0.65, zorder=3)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, color="#AAAAAA")
        ax.set_ylabel("Distance cosinus (shift stylistique)", color="#AAAAAA")
        ax.set_title(title, color="white", pad=12)
        ax.tick_params(colors="#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(axis="y", alpha=0.1, color="#AAAAAA")
        return fig


__all__ = ["StyleAnalyzer", "FUNCTION_WORDS_FR", "PALETTE"]
