#!/usr/bin/env python3
"""
run_best_model.py — Final BERTopic run with nr_topics=20 + outlier reduction.

Fits BERTopic, reduces outliers via embedding-based reassignment,
updates the model, and saves all outputs.
"""

import json
from collections import Counter
from pathlib import Path

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# ── Constants ─────────────────────────────────────────────────────
INPUT_FILE = Path("manifestos_cleaned/all_chunks.json")
OUTPUT_DIR = Path("topic_model_output")
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
NR_TOPICS = 20
MIN_TOPIC_SIZE = 4
PARTY_ORDER = ["BJP", "INC", "CPIM", "TMC", "NCPSP", "DMK", "SP", "NCPAP"]


def main():
    # ── Load data ─────────────────────────────────────────────────
    with open(INPUT_FILE, encoding="utf-8") as f:
        records = json.load(f)

    texts = [r["text"] for r in records]
    parties = [r["party"] for r in records]
    chunk_ids = [r["chunk_id"] for r in records]
    print(f"Loaded {len(texts)} chunks from {INPUT_FILE}\n")

    # ── Embedding ─────────────────────────────────────────────────
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Computing embeddings ...")
    embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}\n")

    # ── Fit BERTopic ──────────────────────────────────────────────
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    print(f"Fitting BERTopic (nr_topics={NR_TOPICS}, "
          f"min_topic_size={MIN_TOPIC_SIZE}) ...")
    model = BERTopic(
        embedding_model=embed_model,
        vectorizer_model=vectorizer,
        nr_topics=NR_TOPICS,
        min_topic_size=MIN_TOPIC_SIZE,
        language="multilingual",
        verbose=True,
    )

    topics, probs = model.fit_transform(texts, embeddings=embeddings)

    n_topics_before = len(set(topics)) - (1 if -1 in topics else 0)
    outliers_before = sum(1 for t in topics if t == -1)
    print(f"\nBefore outlier reduction: {n_topics_before} topics, "
          f"{outliers_before} outliers ({outliers_before/len(topics)*100:.1f}%)\n")

    # ── Outlier reduction ─────────────────────────────────────────
    print("Reducing outliers (strategy='embeddings') ...")
    new_topics = model.reduce_outliers(
        texts, topics, strategy="embeddings", embeddings=embeddings
    )

    outliers_after = sum(1 for t in new_topics if t == -1)
    print(f"After outlier reduction: "
          f"{outliers_after} outliers ({outliers_after/len(new_topics)*100:.1f}%)")

    # Update the model with reassigned topics
    print("Updating model with new topic assignments ...")
    model.update_topics(texts, topics=new_topics, vectorizer_model=vectorizer)
    topics = new_topics

    n_topics_final = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"Final: {n_topics_final} topics\n")

    # ── Topic report ──────────────────────────────────────────────
    sep = "=" * 95
    print(sep)
    print("TOPIC REPORT (after outlier reduction)")
    print(sep)

    for tid in sorted(set(topics)):
        idxs = [i for i, t in enumerate(topics) if t == tid]
        count = len(idxs)

        if tid == -1:
            print(f"\nTopic  -1  |  {count} chunks  (remaining outliers)")
            continue

        kw_tuples = model.get_topic(tid)
        top_kw = [w for w, _ in kw_tuples[:10]]

        party_counter = Counter(parties[i] for i in idxs)
        party_pcts = []
        for p in PARTY_ORDER:
            n = party_counter.get(p, 0)
            if n > 0:
                party_pcts.append(f"{p} {n/count*100:.0f}%")

        print(f"\nTopic {tid:3d}  |  {count} chunks  |  "
              f"Parties: {', '.join(party_pcts)}")
        print(f"  Keywords: {', '.join(top_kw)}")

    print(f"\n{sep}\n")

    # ── Save outputs ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = OUTPUT_DIR / "bertopic_model_20"
    model.save(str(model_path), serialization="safetensors",
               save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL_NAME)
    print(f"Saved model            → {model_path}")

    # chunk_topics_20.json
    doc_records = []
    for cid, party, text, tid in zip(chunk_ids, parties, texts, topics):
        kw = [w for w, _ in model.get_topic(tid)[:10]] if tid != -1 else []
        doc_records.append({
            "chunk_id": cid,
            "party": party,
            "text": text,
            "topic_id": int(tid),
            "topic_keywords": kw,
        })

    ct_path = OUTPUT_DIR / "chunk_topics_20.json"
    with open(ct_path, "w", encoding="utf-8") as f:
        json.dump(doc_records, f, ensure_ascii=False, indent=2)
    print(f"Saved chunk topics     → {ct_path}")

    # Visualisations
    try:
        model.visualize_topics().write_html(
            str(OUTPUT_DIR / "topics_visualization_20.html"))
        print(f"Saved topic viz        → {OUTPUT_DIR / 'topics_visualization_20.html'}")
    except Exception as e:
        print(f"⚠ topics_visualization: {e}")

    try:
        model.visualize_barchart(top_n_topics=NR_TOPICS, n_words=10).write_html(
            str(OUTPUT_DIR / "topic_barchart_20.html"))
        print(f"Saved barchart viz     → {OUTPUT_DIR / 'topic_barchart_20.html'}")
    except Exception as e:
        print(f"⚠ topic_barchart: {e}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"Total topics:                {n_topics_final}")
    print(f"Outliers BEFORE reduction:   {outliers_before}  "
          f"({outliers_before/len(texts)*100:.1f}%)")
    print(f"Outliers AFTER reduction:    {outliers_after}  "
          f"({outliers_after/len(texts)*100:.1f}%)")
    print(f"Chunks reassigned:           {outliers_before - outliers_after}")

    # Top 10 topics by chunk count
    topic_counts = Counter(t for t in topics if t != -1)
    print(f"\nTop 10 topics by size:")
    for tid, cnt in topic_counts.most_common(10):
        kw = [w for w, _ in model.get_topic(tid)[:5]]
        print(f"  Topic {tid:3d} ({cnt:3d} chunks): {', '.join(kw)}")

    print(sep)


if __name__ == "__main__":
    main()
