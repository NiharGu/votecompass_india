#!/usr/bin/env python3
"""
stance_embedding.py — Embedding-based stance scoring with contrastive sentence selection.

Approach:
  1. Extract the single most contrastive sentence from each pole description.
  2. Embed chunks with paraphrase-multilingual-MiniLM-L12-v2.
  3. Score each chunk per axis:
       score = (sim_right - sim_left) / (sim_right + sim_left + 1e-8)
  4. Aggregate per party (mean), flag low-confidence (<3 chunks).
  5. Per-axis min-max normalise to [-1, +1].

Outputs (all new, no existing files touched):
  stance_output/party_vectors_embedding.json
  stance_output/embedding_diagnostics.json
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from axes import AXES, PARTY_ORDER

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNKS_FILE      = Path("manifestos_cleaned/all_chunks.json")
TOPICS_FILE      = Path("topic_model_output/chunk_topics_20.json")
OUTPUT_DIR       = Path("stance_output")
VECTORS_FILE     = OUTPUT_DIR / "party_vectors_embedding.json"
DIAGNOSTICS_FILE = OUTPUT_DIR / "embedding_diagnostics.json"

MODEL_NAME          = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_CHUNKS          = 3     # below this → low_confidence flag
LOW_VAR_THRESHOLD   = 0.15  # std dev below this → flagged in variance report


# ── Sentence utilities ─────────────────────────────────────────────────────────
def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 25]


def most_contrastive_sentence(pole_text: str, other_pole_text: str, model) -> str:
    """
    Return the sentence from pole_text that is MOST DISSIMILAR to other_pole_text.
    Dissimilarity = lowest cosine similarity to the other pole's full embedding.
    """
    sentences = split_sentences(pole_text)
    if len(sentences) <= 1:
        return sentences[0] if sentences else pole_text

    other_emb   = model.encode([other_pole_text], normalize_embeddings=True)
    sent_embs   = model.encode(sentences, normalize_embeddings=True,
                               show_progress_bar=False)
    sims        = cosine_similarity(sent_embs, other_emb).flatten()
    return sentences[int(np.argmin(sims))]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_chunks():
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    # Inject topic_id from chunk_topics file
    with open(TOPICS_FILE, encoding="utf-8") as f:
        topic_records = json.load(f)
    tid_map = {r["chunk_id"]: r["topic_id"] for r in topic_records}
    for c in chunks:
        c["topic_id"] = tid_map.get(c["chunk_id"], -1)

    print(f"Loaded {len(chunks)} chunks  |  topic_ids injected from {TOPICS_FILE}\n")
    return chunks


# ── Chunk selection (hybrid: topic-first, keyword fallback for thin parties) ───
def get_relevant_indices(chunks, axis):
    topic_set = set(axis["topics"])
    keywords  = [kw.lower() for kw in axis["keywords"]]

    topic_idx     = [i for i, c in enumerate(chunks) if c["topic_id"] in topic_set]
    topic_idx_set = set(topic_idx)
    counts        = Counter(chunks[i]["party"] for i in topic_idx)
    thin          = {p for p in PARTY_ORDER if counts.get(p, 0) < MIN_CHUNKS}

    if not thin:
        return topic_idx

    kw_idx = [
        i for i, c in enumerate(chunks)
        if i not in topic_idx_set
        and c["party"] in thin
        and any(kw in c["text"].lower() for kw in keywords)
    ]
    return sorted(topic_idx_set | set(kw_idx))


# ── Scoring ────────────────────────────────────────────────────────────────────
def score_axis(chunks, chunk_embs, left_emb, right_emb, axis):
    rel = get_relevant_indices(chunks, axis)
    if not rel:
        return {}

    r_embs    = chunk_embs[rel]
    sim_l     = cosine_similarity(r_embs, left_emb).flatten()
    sim_r     = cosine_similarity(r_embs, right_emb).flatten()
    scores    = (sim_r - sim_l) / (sim_r + sim_l + 1e-8)

    party_scores = defaultdict(list)
    for j, gi in enumerate(rel):
        party_scores[chunks[gi]["party"]].append(float(scores[j]))
    return dict(party_scores)


# ── Normalisation ──────────────────────────────────────────────────────────────
def minmax_normalise(values: dict):
    """Scale dict of party->float to [-1, +1]. Skips None values."""
    valid = {p: v for p, v in values.items() if v is not None}
    if len(valid) < 2 or len(set(valid.values())) == 1:
        return {p: (0.0 if v is not None else None) for p, v in values.items()}

    lo, hi = min(valid.values()), max(valid.values())
    span   = hi - lo
    return {
        p: (round(2.0 * (v - lo) / span - 1.0, 4) if v is not None else None)
        for p, v in values.items()
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    chunks = load_chunks()

    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # ── Step 1: Extract contrastive sentences ──────────────────────────────────
    print("\n" + "=" * 80)
    print("SELECTED CONTRASTIVE SENTENCES PER AXIS")
    print("=" * 80)

    pole_sents = {}
    for ax in AXES:
        l = most_contrastive_sentence(ax["left_pole"],  ax["right_pole"], model)
        r = most_contrastive_sentence(ax["right_pole"], ax["left_pole"],  model)
        pole_sents[ax["id"]] = {"left": l, "right": r}

        print(f"\n[{ax['id']}]  {ax['label']}")
        print(f"  LEFT  → {l}")
        print(f"  RIGHT → {r}")

    print("\n" + "=" * 80 + "\n")

    # ── Step 2: Encode chunks and poles ───────────────────────────────────────
    print("Encoding chunk embeddings ...")
    chunk_embs = model.encode(
        [c["text"] for c in chunks],
        show_progress_bar=True, batch_size=64, normalize_embeddings=True
    )

    print("\nEncoding pole embeddings ...")
    pole_embs = {
        ax["id"]: {
            "left":  model.encode([pole_sents[ax["id"]]["left"]],  normalize_embeddings=True),
            "right": model.encode([pole_sents[ax["id"]]["right"]], normalize_embeddings=True),
        }
        for ax in AXES
    }

    # ── Step 3: Score all axes ────────────────────────────────────────────────
    print("\nScoring axes ...\n")
    raw_results = {}
    for ax in AXES:
        pd = score_axis(chunks, chunk_embs,
                        pole_embs[ax["id"]]["left"],
                        pole_embs[ax["id"]]["right"], ax)
        raw_results[ax["id"]] = pd
        n_chunks   = sum(len(v) for v in pd.values())
        n_covered  = sum(1 for v in pd.values() if len(v) >= MIN_CHUNKS)
        print(f"  [{ax['id']:20s}]  {n_chunks:4d} chunks  |  {n_covered}/8 parties (≥{MIN_CHUNKS})")

    # ── Step 4: Aggregate + diagnostics ──────────────────────────────────────
    diagnostics    = {}
    raw_means      = {}   # ax_id -> party -> float|None

    for ax in AXES:
        ax_id = ax["id"]
        diagnostics[ax_id] = {}
        raw_means[ax_id]   = {}
        for party in PARTY_ORDER:
            scores = raw_results[ax_id].get(party, [])
            n      = len(scores)
            mean   = float(np.mean(scores)) if n else None
            diagnostics[ax_id][party] = {
                "raw_mean":       round(mean, 4) if mean is not None else None,
                "n_chunks":       n,
                "low_confidence": n < MIN_CHUNKS,
            }
            raw_means[ax_id][party] = mean

    # ── Step 5: Normalise ─────────────────────────────────────────────────────
    normalised = {ax["id"]: minmax_normalise(raw_means[ax["id"]]) for ax in AXES}

    # ── Build party vectors ───────────────────────────────────────────────────
    party_vectors = {
        party: {ax["id"]: normalised[ax["id"]].get(party) for ax in AXES}
        for party in PARTY_ORDER
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(VECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(party_vectors, f, ensure_ascii=False, indent=2)
    with open(DIAGNOSTICS_FILE, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {VECTORS_FILE}")
    print(f"Saved → {DIAGNOSTICS_FILE}")

    # ── Results table ─────────────────────────────────────────────────────────
    abbr  = {"BJP":"BJP","INC":"INC","CPIM":"CPM","TMC":"TMC",
              "NCPSP":"NSP","DMK":"DMK","SP":"SP ","NCPAP":"NAP"}
    abbrs = [abbr[p] for p in PARTY_ORDER]

    print("\n" + "=" * 97)
    print(f"{'AXIS':<32}  " + "  ".join(f"{a:>5}" for a in abbrs))
    print("-" * 97)
    for ax in AXES:
        row = []
        for p in PARTY_ORDER:
            v = normalised[ax["id"]].get(p)
            row.append(" n/a " if v is None else f"{v:+.2f}")
        print(f"{ax['label'][:31]:<32}  " + "  ".join(f"{r:>5}" for r in row))
    print("=" * 97)
    print("\n-1.0 = left pole  |  0.0 = centre  |  +1.0 = right pole")
    print(f"n/a  = party had 0 relevant chunks\n")

    # ── Variance report ───────────────────────────────────────────────────────
    print("VARIANCE REPORT")
    print(f"{'AXIS':<22}  {'STD':>6}  NOTE")
    print("-" * 55)
    for ax in AXES:
        vals = [v for v in normalised[ax["id"]].values() if v is not None]
        std  = float(np.std(vals)) if len(vals) >= 2 else 0.0
        flag = "  ⚠ LOW VARIANCE — needs attention" if std < LOW_VAR_THRESHOLD else ""
        print(f"  {ax['id']:<20}  {std:>6.3f}{flag}")
    print()


if __name__ == "__main__":
    main()
