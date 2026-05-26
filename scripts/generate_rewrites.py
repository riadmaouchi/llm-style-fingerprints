"""
Génère les vraies réécritures LLM pour chaque texte du corpus.

Usage
-----
    python scripts/generate_rewrites.py                     # P1 + P2, tous modèles
    python scripts/generate_rewrites.py --prompt p1         # P1 uniquement
    python scripts/generate_rewrites.py --prompt p2         # P2 uniquement
    python scripts/generate_rewrites.py --model gpt4        # un seul modèle
    python scripts/generate_rewrites.py --fill-pending      # remplit seulement les [PENDING_API]

Clés API requises (dans .env à la racine du projet) :
    OPEN_AI_KEY    — GPT-4
    MISTRAL_KEY    — Mistral 7B
    GEMINI_KEY     — Gemini Pro

Produit
-------
    data/gpt4/rewrites.json      (P1 — 80 textes)
    data/gpt4/rewrites_p2.json   (P2 — 80 textes)
    data/mistral/rewrites.json
    data/mistral/rewrites_p2.json
    data/gemini/rewrites.json
    data/gemini/rewrites_p2.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
import os

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

PROMPTS = {
    "p1": "Réécris ce texte dans un style neutre et factuel, en conservant le sens.",
    "p2": "Reformule ce texte en simplifiant le vocabulaire, pour le rendre accessible au grand public.",
}

SYSTEM = "Tu es un assistant qui réécrit des textes littéraires français."

# Combined corpus positional index: Zola texts first (IDs 1-40),
# then Maupassant texts (IDs 26-65) — must match load_originals() order.
def load_originals() -> list[dict]:
    """Retourne les textes dans l'ordre du corpus combiné (Zola puis Maupassant),
    avec un original_id séquentiel 1-80 correspondant à la position combinée."""
    texts = []
    # Zola: IDs 1-40 → combined positions 1-40
    zola = json.loads((ROOT / "data" / "human" / "zola.json").read_text(encoding="utf-8"))
    for entry in zola["texts"]:
        texts.append({"combined_id": len(texts) + 1, "title": entry["title"], "text": entry["text"]})
    # Maupassant: IDs 26-65 → combined positions 41-80
    maupas = json.loads((ROOT / "data" / "human" / "maupassant.json").read_text(encoding="utf-8"))
    for entry in maupas["texts"]:
        texts.append({"combined_id": len(texts) + 1, "title": entry["title"], "text": entry["text"]})
    return texts


# ---------------------------------------------------------------------------
# GPT-4
# ---------------------------------------------------------------------------

def rewrite_openai(texts: list[dict], prompt: str) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])
    rewrites = []
    for t in texts:
        print(f"  GPT-4  [{t['combined_id']:02d}] {t['title']}")
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"{prompt}\n\n{t['text']}"},
                    ],
                    temperature=0.7,
                    max_tokens=400,
                )
                rewrites.append({
                    "original_id": t["combined_id"],
                    "text": resp.choices[0].message.content.strip(),
                })
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"    retry {attempt+1} after {wait}s: {e}")
                time.sleep(wait)
        time.sleep(0.5)
    return rewrites


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------

def rewrite_mistral(texts: list[dict], prompt: str) -> list[dict]:
    from mistralai.client.sdk import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_KEY"])
    rewrites = []
    for t in texts:
        print(f"  Mistral [{t['combined_id']:02d}] {t['title']}")
        for attempt in range(5):
            try:
                resp = client.chat.complete(
                    model="mistral-small-latest",
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"{prompt}\n\n{t['text']}"},
                    ],
                    temperature=0.7,
                    max_tokens=400,
                )
                rewrites.append({
                    "original_id": t["combined_id"],
                    "text": resp.choices[0].message.content.strip(),
                })
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"    retry {attempt+1} after {wait}s: {e}")
                time.sleep(wait)
        time.sleep(0.5)
    return rewrites


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def rewrite_gemini(texts: list[dict], prompt: str) -> list[dict]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    rewrites = []
    for t in texts:
        print(f"  Gemini  [{t['combined_id']:02d}] {t['title']}")
        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"{SYSTEM}\n\nRéponds uniquement avec le texte réécrit, sans préambule.\n\n{prompt}\n\n{t['text']}",
                    config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1500),
                )
                rewrites.append({
                    "original_id": t["combined_id"],
                    "text": resp.text.strip(),
                })
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"    retry {attempt+1} after {wait}s: {e}")
                time.sleep(wait)
        time.sleep(0.5)
    return rewrites


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(rewrites: list[dict], model_label: str, model_version: str,
         prompt_text: str, prompt_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if prompt_id == "p1" else f"_{prompt_id}"
    payload = {
        "model": model_label,
        "model_version": model_version,
        "prompt": prompt_text,
        "prompt_id": prompt_id,
        "temperature": 0.7,
        "collection_date": "2026-05",
        "rewrites": rewrites,
    }
    path = out_dir / f"rewrites{suffix}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {path} ({len(rewrites)} réécritures)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODELS = {
    "gpt4":   ("GPT-4",      "gpt-4o",              rewrite_openai),
    "mistral": ("Mistral 7B", "mistral-small-latest", rewrite_mistral),
    "gemini":  ("Gemini Pro", "gemini-2.5-flash",     rewrite_gemini),
}


PENDING = "[PENDING_API]"


def fill_pending(model_key: str, prompt_id: str, all_texts: list[dict],
                 fn, label: str, version: str) -> None:
    """Remplace uniquement les entrées [PENDING_API] dans le fichier existant."""
    suffix = "" if prompt_id == "p1" else f"_{prompt_id}"
    path = ROOT / "data" / model_key / f"rewrites{suffix}.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    # Indexer les textes sources par combined_id pour un accès O(1)
    src_by_id = {t["combined_id"]: t for t in all_texts}

    pending_count = sum(1 for e in data["rewrites"] if e["text"] == PENDING)
    if pending_count == 0:
        print(f"  {label} {prompt_id.upper()}: aucun pending, rien à faire.")
        return

    print(f"  {label} {prompt_id.upper()}: {pending_count} entrées à remplir...")
    prompt_text = PROMPTS[prompt_id]

    for entry in data["rewrites"]:
        if entry["text"] != PENDING:
            continue
        oid = entry["original_id"]
        src = src_by_id.get(oid)
        if src is None:
            print(f"    ⚠ original_id={oid} introuvable dans le corpus")
            continue
        # Appel API via la fonction du modèle (attend une liste, retourne une liste)
        result = fn([src], prompt_text)
        if result:
            entry["text"] = result[0]["text"]

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    filled = sum(1 for e in data["rewrites"] if e["text"] != PENDING)
    print(f"  → {path} ({filled}/{len(data['rewrites'])} entrées remplies)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=["p1", "p2", "both"], default="both",
                        help="Prompt(s) à générer (défaut : both)")
    parser.add_argument("--model", choices=list(MODELS), default=None,
                        help="Modèle unique (défaut : tous)")
    parser.add_argument("--fill-pending", action="store_true",
                        help="Remplace uniquement les [PENDING_API] sans toucher aux entrées valides")
    args = parser.parse_args()

    texts = load_originals()
    print(f"Corpus : {len(texts)} textes (Zola + Maupassant)\n")

    prompt_ids = ["p1", "p2"] if args.prompt == "both" else [args.prompt]
    model_keys = [args.model] if args.model else list(MODELS)

    for model_key in model_keys:
        label, version, fn = MODELS[model_key]
        for pid in prompt_ids:
            if args.fill_pending:
                fill_pending(model_key, pid, texts, fn, label, version)
            else:
                prompt_text = PROMPTS[pid]
                print(f"\n{label} — {pid.upper()} ({prompt_text[:50]}...)")
                rewrites = fn(texts, prompt_text)
                save(rewrites, label, version, prompt_text, pid, ROOT / "data" / model_key)

    print("\nTerminé. Lancez 'make fast' pour régénérer les figures.")


if __name__ == "__main__":
    main()
