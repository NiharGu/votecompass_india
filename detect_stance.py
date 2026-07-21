#!/usr/bin/env python3
"""
detect_stance.py — Stage 3c: Stance detection across 13 political axes.

For each axis in axes.py:
  1. For each relevant chunk, run NLI (Natural Language Inference) using
     cross-encoder/nli-deberta-v3-base against both the left_pole and
     right_pole descriptions.
  2. NLI returns three logits: [contradiction, entailment, neutral].
     stance_score = softmax(entailment | right_pole)
                  - softmax(entailment | left_pole)
     Range: approx [-1.0, +1.0]. Positive = agrees with right pole.
  3. Aggregate per party: weighted mean of chunk scores (weights = |score|).
  4. Output an 8x13 stance matrix to topic_model_output/stance_scores.json.

Requires: all_chunks_en.json (run translate_sp.py first to get English SP chunks).
Model: cross-encoder/nli-deberta-v3-base (English NLI, state-of-the-art).
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

from axes import AXES, PARTY_ORDER

# -- Constants -----------------------------------------------------------------
CHUNKS_FILE   = Path("manifestos_cleaned/all_chunks_en.json")  # translated SP
TOPICS_FILE   = Path("topic_model_output/chunk_topics_20.json") # has topic_id
OUTPUT_FILE   = Path("topic_model_output/stance_scores.json")
NLI_MODEL     = "cross-encoder/nli-deberta-v3-base"
BATCH_SIZE    = 16   # NLI pairs per batch; reduce to 8 if OOM on CPU

# Minimum relevant chunks for a party on an axis to produce a score.
MIN_CHUNKS = 2

# NLI label order for cross-encoder/nli-deberta-v3-base
# Model outputs logits in order: [contradiction, entailment, neutral]
LABEL_CONTRADICTION = 0
LABEL_ENTAILMENT    = 1
LABEL_NEUTRAL       = 2


# -- Load data ----------------------------------------------------------------
def load_chunks(chunks_path: Path, topics_path: Path):
    # Load topic assignments (has topic_id per chunk_id)
    with open(topics_path, encoding="utf-8") as f:
        topic_records = json.load(f)
    topic_id_map = {r["chunk_id"]: r["topic_id"] for r in topic_records}

    # Load translated chunks (has English text for SP)
    with open(chunks_path, encoding="utf-8") as f:
        records = json.load(f)

    # Inject topic_id into each chunk
    for c in records:
        c["topic_id"] = topic_id_map.get(c["chunk_id"], -1)

    print(f"Loaded {len(records)} chunks from {chunks_path}")
    sp_count  = sum(1 for c in records if c["party"] == "SP")
    translated = sum(1 for c in records if c.get("translated"))
    print(f"  SP chunks: {sp_count}  |  Translated: {translated}\n")
    return records


# -- Load NLI model -----------------------------------------------------------
def load_nli_model():
    print(f"Loading NLI model: {NLI_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Using device: {device}\n")
    return tokenizer, model, device


# -- NLI entailment score for a list of (premise, hypothesis) pairs -----------
def entailment_scores(pairs, tokenizer, model, device):
    """
    Returns P(entailment) for each (premise, hypothesis) pair.
    Processes in batches of BATCH_SIZE.
    """
    all_probs = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        premises    = [p for p, _ in batch]
        hypotheses  = [h for _, h in batch]

        enc = tokenizer(
            premises, hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits          # (B, 3)
        probs = softmax(logits, dim=-1).cpu()     # (B, 3)
        all_probs.append(probs[:, LABEL_ENTAILMENT])  # P(entailment)

    return torch.cat(all_probs).numpy()           # (N,)


# ── Hybrid chunk selection ────────────────────────────────────────
def get_relevant_indices(chunks, axis):
    """
    Hybrid approach: topic-first, keyword fallback only for thin parties.

    Step 1: Select all chunks whose BERTopic topic_id is in axis["topics"].
            These are high-confidence, topic-matched chunks.

    Step 2: For each party that still has fewer than MIN_CHUNKS topic-matched
            chunks, add keyword-matched chunks from that party only.
            This rescues axes where BERTopic didn't assign enough chunks
            for a specific party, without polluting well-covered parties
            with irrelevant keyword matches.
    """
    from collections import Counter

    topic_set = set(axis["topics"])
    keywords  = [kw.lower() for kw in axis["keywords"]]

    # Step 1 — topic-matched indices
    topic_indices = [i for i, c in enumerate(chunks) if c["topic_id"] in topic_set]
    topic_index_set = set(topic_indices)

    # Count how many topic-matched chunks each party already has
    party_topic_counts = Counter(chunks[i]["party"] for i in topic_indices)
    thin_parties = {p for p in PARTY_ORDER
                    if party_topic_counts.get(p, 0) < MIN_CHUNKS}

    if not thin_parties:
        return topic_indices  # all parties have enough — no fallback needed

    # Step 2 — keyword fallback only for thin parties
    keyword_indices = []
    for i, chunk in enumerate(chunks):
        if i in topic_index_set:
            continue  # already included
        if chunk["party"] not in thin_parties:
            continue  # this party has enough topic chunks
        text_lower = chunk["text"].lower()
        if any(kw in text_lower for kw in keywords):
            keyword_indices.append(i)

    return sorted(topic_index_set | set(keyword_indices))


# -- Score one axis with NLI --------------------------------------------------
def score_axis(axis, chunks, tokenizer, model, device):
    """
    Returns dict: party -> {"score": float|None, "n_chunks": int,
                             "mean_abs": float, "chunk_scores": list}
    """
    rel_idx = get_relevant_indices(chunks, axis)
    if not rel_idx:
        return {p: {"score": None, "n_chunks": 0,
                    "mean_abs": None, "chunk_ids": [],
                    "chunk_scores": []} for p in PARTY_ORDER}

    rel_texts  = [chunks[i]["text"] for i in rel_idx]
    # Use short NLI hypothesis if available, fall back to full pole
    left_pole  = axis.get("left_pole_nli",  axis["left_pole"])
    right_pole = axis.get("right_pole_nli", axis["right_pole"])

    # Build (premise, hypothesis) pairs
    left_pairs  = [(text, left_pole)  for text in rel_texts]
    right_pairs = [(text, right_pole) for text in rel_texts]

    p_left  = entailment_scores(left_pairs,  tokenizer, model, device)  # (M,)
    p_right = entailment_scores(right_pairs, tokenizer, model, device)  # (M,)
    raw_scores = p_right - p_left                                        # (M,)

    # Group by party
    party_data = defaultdict(lambda: {"scores": [], "chunk_ids": []})
    for j, global_i in enumerate(rel_idx):
        chunk = chunks[global_i]
        p = chunk["party"]
        party_data[p]["scores"].append(float(raw_scores[j]))
        party_data[p]["chunk_ids"].append(chunk["chunk_id"])

    results = {}
    for party in PARTY_ORDER:
        pd = party_data.get(party)
        if pd is None or len(pd["scores"]) < MIN_CHUNKS:
            results[party] = {
                "score": None,
                "n_chunks": len(pd["scores"]) if pd else 0,
                "mean_abs": None,
                "chunk_ids": pd["chunk_ids"] if pd else [],
                "chunk_scores": pd["scores"] if pd else [],
            }
            continue

        scores = np.array(pd["scores"])
        weights = np.abs(scores)
        weighted_mean = (
            float(np.average(scores, weights=weights))
            if weights.sum() > 0 else 0.0
        )
        results[party] = {
            "score": round(weighted_mean, 4),
            "n_chunks": len(scores),
            "mean_abs": round(float(np.mean(weights)), 4),
            "chunk_ids": pd["chunk_ids"],
            "chunk_scores": [round(s, 4) for s in scores],
        }
    return results


# ── Pretty-print summary table ────────────────────────────────────
def print_summary(all_scores, axes):
    party_abbr = {
        "BJP": "BJP", "INC": "INC", "CPIM": "CPM",
        "TMC": "TMC", "NCPSP": "NSP", "DMK": "DMK",
        "SP": "SP ", "NCPAP": "NAP",
    }
    abbrs = [party_abbr[p] for p in PARTY_ORDER]

    sep = "=" * 95
    print(sep)
    print(f"{'AXIS':<32}  " + "  ".join(f"{a:>5}" for a in abbrs))
    print("-" * 95)

    for ax in axes:
        ax_scores = all_scores[ax["id"]]
        row_parts = []
        for p in PARTY_ORDER:
            val = ax_scores.get(p, {}).get("score")
            if val is None:
                row_parts.append(" n/a ")
            else:
                row_parts.append(f"{val:+.2f}")
        label = ax["label"][:31]
        print(f"{label:<32}  " + "  ".join(f"{r:>5}" for r in row_parts))

    print(sep)
    print("\nScale: -1.0 = left pole  |  0.0 = centre  |  +1.0 = right pole")
    print("n/a  = fewer than", MIN_CHUNKS, "relevant chunks for that party\n")


# -- Main ---------------------------------------------------------------------
def main():
    chunks = load_chunks(CHUNKS_FILE, TOPICS_FILE)
    tokenizer, nli_model, device = load_nli_model()

    all_scores = {}
    print("Scoring axes with NLI ...\n")
    for ax_idx, ax in enumerate(AXES):
        print(f"  [{ax_idx+1:2d}/13] {ax['label']} ...", end=" ", flush=True)
        result = score_axis(ax, chunks, tokenizer, nli_model, device)
        all_scores[ax["id"]] = result
        n_rel = sum(d["n_chunks"] for d in result.values())
        scored = sum(1 for d in result.values() if d["score"] is not None)
        print(f"{n_rel} chunks, {scored}/8 parties scored")

    print()
    print_summary(all_scores, AXES)

    # -- Save ------------------------------------------------------------------
    output = {
        "model": NLI_MODEL,
        "method": "nli_entailment",
        "axes": [ax["id"] for ax in AXES],
        "parties": PARTY_ORDER,
        "scores": all_scores,
        "matrix": {
            ax["id"]: {
                p: all_scores[ax["id"]].get(p, {}).get("score")
                for p in PARTY_ORDER
            }
            for ax in AXES
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved stance scores → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
