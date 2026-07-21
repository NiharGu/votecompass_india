#!/usr/bin/env python3
"""
tune_topics.py — Evaluate BERTopic across multiple nr_topics values.

Pre-computes embeddings once, then sweeps nr_topics = [10,15,20,25,30,35,40],
scoring each on outlier rate, coherence (c_v), topic diversity, and noise
topic rate.  Saves the best model and a results CSV.
"""

import json
import csv
import time
from collections import Counter
from pathlib import Path

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# ── Constants ─────────────────────────────────────────────────────
INPUT_FILE = Path("manifestos_cleaned/all_chunks.json")
OUTPUT_DIR = Path("topic_model_output")
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
NR_TOPICS_GRID = [10, 15, 20, 25, 30, 35, 40]
MIN_TOPIC_SIZE = 4
TOP_N_WORDS = 10

PARTY_ORDER = ["BJP", "INC", "CPIM", "TMC", "NCPSP", "DMK", "SP", "NCPAP"]


# ── Load data ─────────────────────────────────────────────────────
def load_data():
    with open(INPUT_FILE, encoding="utf-8") as f:
        records = json.load(f)
    texts = [r["text"] for r in records]
    parties = [r["party"] for r in records]
    chunk_ids = [r["chunk_id"] for r in records]
    print(f"Loaded {len(texts)} chunks from {INPUT_FILE}")
    return texts, parties, chunk_ids


# ── Pre-compute embeddings ────────────────────────────────────────
def compute_embeddings(texts):
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Computing embeddings (one-time) ...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}\n")
    return model, embeddings


# ── Metrics ───────────────────────────────────────────────────────

def metric_outlier_rate(topics):
    """Percentage of chunks assigned to topic -1."""
    n_outliers = sum(1 for t in topics if t == -1)
    return n_outliers / len(topics)


def metric_coherence_cv(model, texts):
    """
    Compute c_v coherence using gensim.
    Tokenises texts, builds a gensim Dictionary, and scores each topic's
    top-N keywords against the corpus.
    """
    # Get topic keywords (excluding outlier topic -1)
    topic_words = []
    for tid in model.get_topic_info()["Topic"]:
        if tid == -1:
            continue
        kw = [w for w, _ in model.get_topic(tid)[:TOP_N_WORDS]]
        if kw:
            topic_words.append(kw)

    if not topic_words:
        return 0.0

    # Tokenise the corpus for gensim
    tokenised = [t.lower().split() for t in texts]
    dictionary = Dictionary(tokenised)

    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenised,
        dictionary=dictionary,
        coherence="c_v",
    )
    return cm.get_coherence()


def metric_diversity(model):
    """
    Fraction of unique words across all topic keyword lists.
    1.0 = every topic uses completely different words (ideal).
    """
    all_words = []
    for tid in model.get_topic_info()["Topic"]:
        if tid == -1:
            continue
        kw = [w for w, _ in model.get_topic(tid)[:TOP_N_WORDS]]
        all_words.extend(kw)

    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def metric_noise_topic_rate(model, topics, parties):
    """
    Fraction of topics where a single party contributes > 60% of chunks.
    These are party-specific noise topics, not genuine policy themes.
    """
    topic_ids = [t for t in set(topics) if t != -1]
    if not topic_ids:
        return 1.0

    noise_count = 0
    for tid in topic_ids:
        idxs = [i for i, t in enumerate(topics) if t == tid]
        if not idxs:
            continue
        party_counter = Counter(parties[i] for i in idxs)
        max_pct = max(party_counter.values()) / len(idxs)
        if max_pct > 0.60:
            noise_count += 1

    return noise_count / len(topic_ids)


# ── Main loop ─────────────────────────────────────────────────────
def main():
    texts, parties, chunk_ids = load_data()
    embed_model, embeddings = compute_embeddings(texts)

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    results = []
    best_score = -1
    best_nr = None
    best_model = None
    best_topics = None

    sep = "=" * 90
    print(sep)
    print(f"{'nr_topics':>10} | {'Topics':>6} | {'Outlier%':>8} | "
          f"{'Coherence':>9} | {'Diversity':>9} | {'Noise%':>7} | {'Score':>8} | Time")
    print("-" * 90)

    for nr in NR_TOPICS_GRID:
        t0 = time.time()

        model = BERTopic(
            embedding_model=embed_model,
            vectorizer_model=vectorizer,
            nr_topics=nr,
            min_topic_size=MIN_TOPIC_SIZE,
            language="multilingual",
            verbose=False,
        )

        topics, _ = model.fit_transform(texts, embeddings=embeddings)
        elapsed = time.time() - t0

        n_topics = len(set(topics)) - (1 if -1 in topics else 0)

        # Compute metrics
        outlier = metric_outlier_rate(topics)
        coherence = metric_coherence_cv(model, texts)
        diversity = metric_diversity(model)
        noise = metric_noise_topic_rate(model, topics, parties)

        # Combined score
        score = coherence * diversity * (1 - outlier) * (1 - noise)

        results.append({
            "nr_topics": nr,
            "actual_topics": n_topics,
            "outlier_rate": outlier,
            "coherence_cv": coherence,
            "diversity": diversity,
            "noise_topic_rate": noise,
            "combined_score": score,
        })

        print(f"{nr:>10} | {n_topics:>6} | {outlier:>7.1%} | "
              f"{coherence:>9.4f} | {diversity:>9.4f} | {noise:>6.1%} | "
              f"{score:>8.4f} | {elapsed:.1f}s")

        if score > best_score:
            best_score = score
            best_nr = nr
            best_model = model
            best_topics = topics

    print(sep)
    print(f"\n★  Best configuration: nr_topics={best_nr} "
          f"(combined score = {best_score:.4f})\n")

    # ── Save results CSV ──────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "tuning_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved tuning results → {csv_path}")

    # ── Save best model & outputs ─────────────────────────────────
    # Model
    model_path = OUTPUT_DIR / "bertopic_model"
    best_model.save(str(model_path), serialization="safetensors",
                    save_ctfidf=True,
                    save_embedding_model=EMBEDDING_MODEL_NAME)
    print(f"Saved best model     → {model_path}")

    # topic_info.csv
    best_model.get_topic_info().to_csv(
        OUTPUT_DIR / "topic_info.csv", index=False)
    print(f"Saved topic info     → {OUTPUT_DIR / 'topic_info.csv'}")

    # chunk_topics.json
    doc_records = []
    for i, (cid, party, text, tid) in enumerate(
            zip(chunk_ids, parties, texts, best_topics)):
        kw = [w for w, _ in best_model.get_topic(tid)[:TOP_N_WORDS]] \
            if tid != -1 else []
        doc_records.append({
            "chunk_id": cid, "party": party, "text": text,
            "topic_id": int(tid), "topic_keywords": kw,
        })
    with open(OUTPUT_DIR / "chunk_topics.json", "w", encoding="utf-8") as f:
        json.dump(doc_records, f, ensure_ascii=False, indent=2)
    print(f"Saved chunk topics   → {OUTPUT_DIR / 'chunk_topics.json'}")

    # Visualisations
    try:
        best_model.visualize_topics().write_html(
            str(OUTPUT_DIR / "topics_visualization.html"))
        print(f"Saved topic viz      → {OUTPUT_DIR / 'topics_visualization.html'}")
    except Exception as e:
        print(f"⚠ topics_visualization: {e}")

    try:
        best_model.visualize_barchart(
            top_n_topics=best_nr, n_words=10).write_html(
            str(OUTPUT_DIR / "topic_barchart.html"))
        print(f"Saved barchart viz   → {OUTPUT_DIR / 'topic_barchart.html'}")
    except Exception as e:
        print(f"⚠ topic_barchart: {e}")

    # ── Print best model topic summary ────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"BEST MODEL TOPIC SUMMARY  (nr_topics={best_nr})")
    print(f"{'=' * 70}")
    for tid in sorted(set(best_topics)):
        idxs = [i for i, t in enumerate(best_topics) if t == tid]
        count = len(idxs)
        if tid == -1:
            print(f"\n  Topic -1  |  {count} chunks (outliers)")
            continue
        kw = [w for w, _ in best_model.get_topic(tid)[:8]]
        party_counter = Counter(parties[i] for i in idxs)
        top_parties = ", ".join(
            f"{p} {party_counter[p]/count*100:.0f}%"
            for p in PARTY_ORDER if party_counter.get(p, 0) > 0
        )
        print(f"\n  Topic {tid:2d}  |  {count} chunks  |  {top_parties}")
        print(f"    Keywords: {', '.join(kw)}")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
