#!/usr/bin/env python3
"""
detect_stance_cosine_fix1.py — Quick diagnostic run.

Same cosine-similarity scorer as the original detect_stance.py, but with
Fix 1 applied: chunk selection uses ONLY BERTopic topic_id matching.
No keyword fallback. This shows the isolated effect of Fix 1.

Output: topic_model_output/stance_scores_cosine_fix1.json
        (does not overwrite the NLI run in progress)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from axes import AXES, PARTY_ORDER

CHUNKS_FILE  = Path("topic_model_output/chunk_topics_20.json")  # has topic_id
OUTPUT_FILE  = Path("topic_model_output/stance_scores_cosine_fix1.json")
MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_CHUNKS   = 2


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def embed_all(model, chunks, axes):
    texts = [c["text"] for c in chunks]
    print("Encoding chunk embeddings ...")
    chunk_embs = model.encode(texts, show_progress_bar=True,
                              batch_size=64, normalize_embeddings=True)
    print("Encoding pole embeddings ...")
    pole_embs = {}
    for ax in axes:
        pole_embs[ax["id"]] = {
            "left":  model.encode([ax["left_pole"]],  normalize_embeddings=True),
            "right": model.encode([ax["right_pole"]], normalize_embeddings=True),
        }
    print(f"Done — {len(axes)*2} pole vectors.\n")
    return chunk_embs, pole_embs


# ── FIX 1: topic-only, no keyword fallback ────────────────────────
def get_relevant_indices(chunks, axis):
    topic_set = set(axis["topics"])
    return [i for i, c in enumerate(chunks) if c["topic_id"] in topic_set]


def score_axis(axis, chunks, chunk_embs, pole_embs):
    rel_idx = get_relevant_indices(chunks, axis)
    if not rel_idx:
        return {p: {"score": None, "n_chunks": 0, "mean_abs": None,
                    "chunk_ids": [], "chunk_scores": []} for p in PARTY_ORDER}

    rel_embs  = chunk_embs[rel_idx]
    sim_left  = cosine_similarity(rel_embs, pole_embs[axis["id"]]["left"]).flatten()
    sim_right = cosine_similarity(rel_embs, pole_embs[axis["id"]]["right"]).flatten()
    raw       = sim_right - sim_left

    party_data = defaultdict(lambda: {"scores": [], "chunk_ids": []})
    for j, gi in enumerate(rel_idx):
        p = chunks[gi]["party"]
        party_data[p]["scores"].append(float(raw[j]))
        party_data[p]["chunk_ids"].append(chunks[gi]["chunk_id"])

    results = {}
    for party in PARTY_ORDER:
        pd = party_data.get(party)
        if pd is None or len(pd["scores"]) < MIN_CHUNKS:
            results[party] = {
                "score": None, "n_chunks": len(pd["scores"]) if pd else 0,
                "mean_abs": None,
                "chunk_ids": pd["chunk_ids"] if pd else [],
                "chunk_scores": pd["scores"] if pd else [],
            }
            continue
        scores  = np.array(pd["scores"])
        weights = np.abs(scores)
        wm = float(np.average(scores, weights=weights)) if weights.sum() > 0 else 0.0
        results[party] = {
            "score": round(wm, 4),
            "n_chunks": len(scores),
            "mean_abs": round(float(np.mean(weights)), 4),
            "chunk_ids": pd["chunk_ids"],
            "chunk_scores": [round(s, 4) for s in scores],
        }
    return results


def print_summary(all_scores):
    abbr = {"BJP":"BJP","INC":"INC","CPIM":"CPM","TMC":"TMC",
            "NCPSP":"NSP","DMK":"DMK","SP":"SP ","NCPAP":"NAP"}
    abbrs = [abbr[p] for p in PARTY_ORDER]
    sep = "=" * 95
    print(sep)
    print(f"{'AXIS':<32}  " + "  ".join(f"{a:>5}" for a in abbrs) + "  | n_chunks")
    print("-" * 95)
    for ax in AXES:
        sc = all_scores[ax["id"]]
        row = []
        for p in PARTY_ORDER:
            v = sc.get(p, {}).get("score")
            row.append(" n/a " if v is None else f"{v:+.2f}")
        n_total = sum(sc.get(p, {}).get("n_chunks", 0) for p in PARTY_ORDER)
        label = ax["label"][:31]
        print(f"{label:<32}  " + "  ".join(f"{r:>5}" for r in row) + f"  | {n_total}")
    print(sep)
    print("\n-1.0 = left pole  |  0.0 = centre  |  +1.0 = right pole")
    print("n/a  = fewer than", MIN_CHUNKS, "topic-matched chunks\n")


def main():
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}\n")

    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    chunk_embs, pole_embs = embed_all(model, chunks, AXES)

    print("Scoring (Fix 1 — topic-only) ...")
    all_scores = {}
    for ax in AXES:
        result = score_axis(ax, chunks, chunk_embs, pole_embs)
        all_scores[ax["id"]] = result
        n_rel    = sum(d["n_chunks"] for d in result.values())
        n_scored = sum(1 for d in result.values() if d["score"] is not None)
        print(f"  [{ax['id']:18s}]  {n_rel:3d} chunks  |  {n_scored}/8 parties")

    print()
    print_summary(all_scores)

    output = {
        "model": MODEL_NAME, "method": "cosine_fix1_topic_only",
        "axes": [ax["id"] for ax in AXES], "parties": PARTY_ORDER,
        "scores": all_scores,
        "matrix": {
            ax["id"]: {p: all_scores[ax["id"]].get(p, {}).get("score")
                       for p in PARTY_ORDER}
            for ax in AXES
        },
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
