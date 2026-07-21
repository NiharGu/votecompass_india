#!/usr/bin/env python3
"""
translate_sp.py — Translate SP (Samajwadi Party) chunks from Hindi to English.

Upgraded from Helsinki-NLP/opus-mt-hi-en (poor quality) to
facebook/nllb-200-distilled-600M, which supports 200 languages with
significantly better quality for Hindi political text.

Output: manifestos_cleaned/all_chunks_en.json
  - All non-SP chunks: unchanged
  - SP chunks: "text" = English translation, "text_original" = original Hindi
"""

import re
import json
from pathlib import Path

INPUT_FILE  = Path("manifestos_cleaned/all_chunks.json")
OUTPUT_FILE = Path("manifestos_cleaned/all_chunks_en.json")

# NLLB-200 language codes
SRC_LANG = "hin_Deva"   # Hindi in Devanagari script
TGT_LANG = "eng_Latn"   # English in Latin script

MODEL_NAME  = "facebook/nllb-200-distilled-600M"
BATCH_SIZE  = 1   # one chunk at a time to handle variable length safely
MAX_TOKENS  = 512


def clean_hindi(text: str) -> str:
    """
    Remove OCR garbage from the SP manifesto text before translation.
    Keeps:
      - Devanagari characters (U+0900–U+097F)
      - Basic ASCII (letters, digits, punctuation, spaces)
      - Common symbols used in manifesto formatting (>, |, -, •)
    Strips:
      - Mixed digit+Devanagari OCR artifacts (e.g. 5९५, ॥5६2९॥)
      - Sequences of mixed scripts that form no valid word
    """
    # Remove runs of characters that mix ASCII digits with Devanagari
    # e.g. "5९५", "5०0५४९/९४॥४५", "#॥धाटां३"
    text = re.sub(r'[0-9#@&]+[\u0900-\u097F॥]+[0-9#@&॥]*', '', text)
    text = re.sub(r'[\u0900-\u097F॥]+[0-9#@&]+[\u0900-\u097F0-9#@&॥]*', '', text)
    # Remove lone ॥ (double danda used as separator in garbled text)
    text = re.sub(r'॥+', ' ', text)
    # Collapse multiple spaces/newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def translate_text(text, tokenizer, model, device):
    """
    Translate a single Hindi text to English.
    Cleans OCR garbage first, then translates.
    """
    import torch
    cleaned = clean_hindi(text)
    if not cleaned:
        return "[No translatable content]"

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        src_lang=SRC_LANG,
    ).to(device)

    target_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    with torch.no_grad():
        translated = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            num_beams=4,
            max_length=MAX_TOKENS,
        )

    return tokenizer.decode(translated[0], skip_special_tokens=True)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    chunks = load_chunks(INPUT_FILE)
    sp_chunks    = [c for c in chunks if c["party"] == "SP"]
    other_chunks = [c for c in chunks if c["party"] != "SP"]

    print(f"Total chunks      : {len(chunks)}")
    print(f"SP chunks (Hindi) : {len(sp_chunks)}")
    print(f"Other chunks      : {len(other_chunks)}")

    if not sp_chunks:
        print("No SP chunks found — nothing to translate.")
        return

    # ── Load NLLB-200 ─────────────────────────────────────────────
    print(f"\nLoading translation model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    # ── Translate in batches ───────────────────────────────────────
    print(f"\nTranslating {len(sp_chunks)} SP chunks one at a time ...\n")
    translated_chunks = []

    for i, chunk in enumerate(sp_chunks):
        translation = translate_text(chunk["text"], tokenizer, model, device)

        new_chunk = dict(chunk)
        new_chunk["text_original"] = chunk["text"]
        new_chunk["text"]          = translation
        new_chunk["translated"]    = True
        translated_chunks.append(new_chunk)
        print(f"  [{i+1:3d}/{len(sp_chunks)}] {chunk['chunk_id']}")

    # ── Spot-check ─────────────────────────────────────────────────
    print("\n── Spot-check (first 3 SP chunks) ──────────────────────────")
    for c in translated_chunks[:3]:
        print(f"\n  ID: {c['chunk_id']}")
        print(f"  HI: {c['text_original'][:140]}")
        print(f"  EN: {c['text'][:140]}")

    # ── Merge and save ─────────────────────────────────────────────
    id_to_translated = {c["chunk_id"]: c for c in translated_chunks}
    merged = []
    for c in chunks:
        if c["party"] == "SP":
            merged.append(id_to_translated[c["chunk_id"]])
        else:
            merged.append(c)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(merged)} chunks → {OUTPUT_FILE}")
    print(f"  SP chunks translated : {len(translated_chunks)}")
    print(f"  Other chunks copied  : {len(other_chunks)}")


if __name__ == "__main__":
    main()
