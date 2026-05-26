"""
src/data.py
~~~~~~~~~~~

Chargement du corpus — source unique partagée entre notebooks et scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

# Racine du projet (deux niveaux au-dessus de ce fichier)
_PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = _PROJECT_ROOT / "data"


class TextEntry(TypedDict):
    id: int
    text: str


class HumanCorpus(TypedDict):
    source: str
    language: str
    texts: list[TextEntry]


class RewriteCorpus(TypedDict):
    model: str
    model_version: str
    prompt: str
    rewrites: list[TextEntry]


# ---------------------------------------------------------------------------
# Loaders bas niveau
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_human(name: str) -> list[str]:
    """Charge les textes d'un auteur humain.

    Paramètre
    ---------
    name : "zola" | "maupassant"

    Retourne
    --------
    liste de strings (textes bruts)
    """
    path = DATA_DIR / "human" / f"{name}.json"
    return [t["text"] for t in _load_json(path)["texts"]]


_PENDING = "[PENDING_API]"


def load_rewrites(model: str, prompt: str = "p1") -> list[str]:
    """Charge les réécritures d'un modèle LLM.

    Paramètres
    ----------
    model  : "gpt4" | "claude3" | "mistral" | "gemini"
    prompt : "p1" (neutre) | "p2" (simplification)

    Retourne
    --------
    liste de strings — les entrées [PENDING_API] restent telles quelles ;
    utilisez load_aligned_rewrites() pour filtrer les textes incomplets.
    """
    suffix = "" if prompt == "p1" else f"_{prompt}"
    path = DATA_DIR / model / f"rewrites{suffix}.json"
    return [r["text"] for r in _load_json(path)["rewrites"]]


def load_aligned_rewrites(prompt: str = "p1") -> tuple[list[str], dict[str, list[str]], list[int]]:
    """Retourne les textes originaux et les réécritures alignés sur le sous-ensemble
    pour lequel TOUS les modèles ont des réécritures disponibles (non-PENDING).

    Retourne
    --------
    (originals, rewrites_dict, valid_indices)
      - originals      : textes sources filtrés
      - rewrites_dict  : {"GPT-4": [...], "Claude 3": [...], ...}
      - valid_indices  : indices dans le corpus complet (utile pour split Zola/Maupas)
    """
    zola = load_human("zola")
    maupas = load_human("maupassant")
    all_originals = zola + maupas
    models = {"GPT-4": "gpt4", "Claude 3": "claude3", "Mistral 7B": "mistral", "Gemini Pro": "gemini"}
    raw = {label: load_rewrites(slug, prompt) for label, slug in models.items()}

    valid_idx = [
        i for i in range(len(all_originals))
        if all(i < len(raw[lbl]) and raw[lbl][i] != _PENDING for lbl in raw)
    ]

    aligned_orig = [all_originals[i] for i in valid_idx]
    aligned_rew = {lbl: [raw[lbl][i] for i in valid_idx] for lbl in raw}
    return aligned_orig, aligned_rew, valid_idx


def load_single_model_aligned(model: str, prompt: str = "p1") -> tuple[list[str], list[str]]:
    """Retourne les textes originaux et réécritures pour UN seul modèle,
    en filtrant seulement ses propres entrées PENDING.

    Utile pour la figure de robustesse inter-prompt (compare p1 vs p2 par modèle).

    Retourne
    --------
    (originals, rewrites) — listes de même longueur
    """
    all_orig = load_human("zola") + load_human("maupassant")
    rewrites = load_rewrites(model, prompt)
    valid_idx = [i for i in range(len(all_orig)) if i < len(rewrites) and rewrites[i] != _PENDING]
    return [all_orig[i] for i in valid_idx], [rewrites[i] for i in valid_idx]


def load_model_meta(model: str) -> dict:
    """Retourne les métadonnées d'un modèle (nom, version, prompt)."""
    path = DATA_DIR / model / "rewrites.json"
    data = _load_json(path)
    return {k: data[k] for k in ("model", "model_version", "prompt") if k in data}


# ---------------------------------------------------------------------------
# Loader haut niveau — corpus complet
# ---------------------------------------------------------------------------

def load_corpus() -> dict[str, list[str]]:
    """
    Charge tout le corpus en un dict label → textes.

    Les textes [PENDING_API] (en attente d'appel API) sont exclus des groupes
    LLM. Les textes humains correspondants sont également filtrés pour que
    toutes les listes soient alignées sur le même sous-ensemble.

    Retourne
    --------
    {
        "Humain (Zola)":       [...],
        "Humain (Maupassant)": [...],
        "GPT-4":               [...],
        "Claude 3":            [...],
        "Mistral 7B":          [...],
        "Gemini Pro":          [...],
    }
    """
    originals, aligned_rew, valid_idx = load_aligned_rewrites()
    n_zola = len(load_human("zola"))
    zola_texts  = [originals[j] for j, i in enumerate(valid_idx) if i < n_zola]
    maupas_texts = [originals[j] for j, i in enumerate(valid_idx) if i >= n_zola]
    return {
        "Humain (Zola)":       zola_texts,
        "Humain (Maupassant)": maupas_texts,
        **aligned_rew,
    }


def load_originals() -> list[str]:
    """Retourne les textes originaux alignés (Zola + Maupassant) excluant
    les positions sans réécriture pour tous les modèles."""
    originals, _, _idx = load_aligned_rewrites()
    return originals


def load_llm_corpora() -> dict[str, list[str]]:
    """Retourne les réécritures LLM alignées (sans les humains, sans PENDING)."""
    _, rewrites, _idx = load_aligned_rewrites()
    return rewrites


__all__ = [
    "DATA_DIR",
    "load_human",
    "load_rewrites",
    "load_aligned_rewrites",
    "load_single_model_aligned",
    "load_model_meta",
    "load_corpus",
    "load_originals",
    "load_llm_corpora",
]
