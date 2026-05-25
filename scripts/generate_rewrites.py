"""
Génère les vraies réécritures LLM pour chaque texte du corpus.

Usage
-----
    python scripts/generate_rewrites.py

Clés API requises (dans .env à la racine du projet) :
    OPEN_AI_KEY    — GPT-4
    MISTRAL_KEY    — Mistral 7B
    GEMINI_KEY     — Gemini Pro

Produit
-------
    data/gpt4/rewrites.json
    data/mistral/rewrites.json
    data/gemini/rewrites.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv
import os

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

PROMPT = "Réécris ce texte dans un style neutre et factuel, en conservant le sens."


def load_originals() -> list[dict]:
    texts = []
    for fname in ["zola.json", "maupassant.json"]:
        data = json.loads((ROOT / "data" / "human" / fname).read_text(encoding="utf-8"))
        texts.extend(data["texts"])
    return texts


# ---------------------------------------------------------------------------
# GPT-4
# ---------------------------------------------------------------------------

def rewrite_openai(texts: list[dict]) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])
    rewrites = []
    for t in texts:
        print(f"  GPT-4  [{t['id']:02d}] {t['title']}")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Tu es un assistant qui réécrit des textes littéraires français."},
                {"role": "user", "content": f"{PROMPT}\n\n{t['text']}"},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        rewrites.append({"id": t["id"], "original_id": t["id"], "text": resp.choices[0].message.content.strip()})
        time.sleep(0.5)
    return rewrites


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------

def rewrite_mistral(texts: list[dict]) -> list[dict]:
    from mistralai import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_KEY"])
    rewrites = []
    for t in texts:
        print(f"  Mistral [{t['id']:02d}] {t['title']}")
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "Tu es un assistant qui réécrit des textes littéraires français."},
                {"role": "user", "content": f"{PROMPT}\n\n{t['text']}"},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        rewrites.append({"id": t["id"], "original_id": t["id"], "text": resp.choices[0].message.content.strip()})
        time.sleep(0.5)
    return rewrites


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def rewrite_gemini(texts: list[dict]) -> list[dict]:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    rewrites = []
    for t in texts:
        print(f"  Gemini  [{t['id']:02d}] {t['title']}")
        resp = model.generate_content(
            f"{PROMPT}\n\n{t['text']}",
            generation_config={"temperature": 0.7, "max_output_tokens": 400},
        )
        rewrites.append({"id": t["id"], "original_id": t["id"], "text": resp.text.strip()})
        time.sleep(0.5)
    return rewrites


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(rewrites: list[dict], model_key: str, model_version: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_key,
        "model_version": model_version,
        "prompt": PROMPT,
        "temperature": 0.7,
        "rewrites": rewrites,
    }
    path = out_dir / "rewrites.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    texts = load_originals()
    print(f"Corpus : {len(texts)} textes\n")

    print("GPT-4o...")
    gpt4_rewrites = rewrite_openai(texts)
    save(gpt4_rewrites, "GPT-4", "gpt-4o", ROOT / "data" / "gpt4")

    print("\nMistral...")
    mistral_rewrites = rewrite_mistral(texts)
    save(mistral_rewrites, "Mistral 7B", "mistral-small-latest", ROOT / "data" / "mistral")

    print("\nGemini...")
    gemini_rewrites = rewrite_gemini(texts)
    save(gemini_rewrites, "Gemini Pro", "gemini-1.5-flash", ROOT / "data" / "gemini")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
