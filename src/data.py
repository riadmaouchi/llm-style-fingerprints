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


def load_rewrites(model: str) -> list[str]:
    """Charge les réécritures d'un modèle LLM.

    Paramètre
    ---------
    model : "gpt4" | "claude3" | "mistral" | "gemini"

    Retourne
    --------
    liste de strings (réécritures, dans le même ordre que les originaux)
    """
    path = DATA_DIR / model / "rewrites.json"
    return [r["text"] for r in _load_json(path)["rewrites"]]


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
    return {
        "Humain (Zola)":       load_human("zola"),
        "Humain (Maupassant)": load_human("maupassant"),
        "GPT-4":               load_rewrites("gpt4"),
        "Claude 3":            load_rewrites("claude3"),
        "Mistral 7B":          load_rewrites("mistral"),
        "Gemini Pro":          load_rewrites("gemini"),
    }


def load_originals() -> list[str]:
    """Retourne les 16 textes originaux (Zola + Maupassant) dans l'ordre."""
    return load_human("zola") + load_human("maupassant")


def load_llm_corpora() -> dict[str, list[str]]:
    """Retourne uniquement les groupes LLM (sans les humains)."""
    return {
        "GPT-4":     load_rewrites("gpt4"),
        "Claude 3":  load_rewrites("claude3"),
        "Mistral 7B": load_rewrites("mistral"),
        "Gemini Pro": load_rewrites("gemini"),
    }


__all__ = [
    "DATA_DIR",
    "load_human",
    "load_rewrites",
    "load_model_meta",
    "load_corpus",
    "load_originals",
    "load_llm_corpora",
]
