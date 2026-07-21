#!/usr/bin/env python3
"""
topic_discovery.py — BERTopic-based topic modelling for political manifestos.

Loads chunked manifesto text (English + Hindi), embeds with a multilingual
sentence-transformer, clusters via BERTopic, and produces a full topic
report with per-party breakdowns plus interactive visualisations.
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ── Step 1: Install & import dependencies ─────────────────────────
REQUIRED = ["bertopic", "sentence-transformers", "umap-learn", "hdbscan"]

def _ensure_deps():
    """Install any missing packages once."""
    missing = []
    for pkg in REQUIRED:
        import_name = pkg.replace("-", "_")  # umap-learn → umap_learn
        if pkg == "umap-learn":
            import_name = "umap"
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
        )
        print("Done.\n")

_ensure_deps()

from bertopic import BERTopic                       # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────
INPUT_FILE = Path("manifestos_cleaned/all_chunks.json")
OUTPUT_DIR = Path("topic_model_output")
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TARGET_TOPICS = 25
MIN_TOPIC_SIZE = 4

PARTY_ORDER = ["BJP", "INC", "CPIM", "TMC", "NCPSP", "DMK", "SP", "NCPAP"]


# ── Step 2: Load chunks ──────────────────────────────────────────
def load_chunks(path: Path):
    """Load chunks JSON and return texts, parties, and full records."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    texts = [r["text"] for r in records]
    parties = [r["party"] for r in records]
    chunk_ids = [r["chunk_id"] for r in records]

    print(f"Loaded {len(texts)} chunks from {path}")
    party_counts = Counter(parties)
    for p in PARTY_ORDER:
        print(f"  {p:6s}: {party_counts.get(p, 0):4d} chunks")
    print()
    return texts, parties, chunk_ids, records


# ── Step 3: Run BERTopic ──────────────────────────────────────────
def run_bertopic(texts):
    """Fit BERTopic on the full corpus and return model + assignments."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Use CountVectorizer to strip English stopwords + enable bigrams
    # for more meaningful topic keywords
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    print(f"Fitting BERTopic (target {TARGET_TOPICS} topics, "
          f"min_topic_size={MIN_TOPIC_SIZE}) ...")
    model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        nr_topics=TARGET_TOPICS,
        min_topic_size=MIN_TOPIC_SIZE,
        language="multilingual",
        verbose=True,
    )

    topics, probs = model.fit_transform(texts)
    print(f"BERTopic fit complete — {len(set(topics)) - (1 if -1 in topics else 0)} "
          f"topics discovered (plus outlier topic -1).\n")
    return model, topics, probs


# ── Step 4: Print topic report ────────────────────────────────────
def print_topic_report(model, topics, parties):
    """Print detailed per-topic report with party breakdown."""
    topic_info = model.get_topic_info()
    unique_topics = sorted(set(topics))

    sep = "=" * 90
    print(sep)
    print("TOPIC REPORT")
    print(sep)

    for tid in unique_topics:
        # Indices of docs assigned to this topic
        doc_idxs = [i for i, t in enumerate(topics) if t == tid]
        count = len(doc_idxs)

        # Top keywords
        if tid == -1:
            keywords_str = "(outliers — no coherent topic)"
        else:
            kw_tuples = model.get_topic(tid)
            top_kw = [w for w, _ in kw_tuples[:10]]
            keywords_str = ", ".join(top_kw)

        # Party breakdown
        party_counter = Counter(parties[i] for i in doc_idxs)
        party_pcts = []
        for p in PARTY_ORDER:
            n = party_counter.get(p, 0)
            if n > 0:
                pct = n / count * 100
                party_pcts.append(f"{p} {pct:.0f}%")

        print(f"\nTopic {tid:3d}  |  {count} chunks  |  "
              f"Parties: {', '.join(party_pcts)}")
        print(f"  Keywords: {keywords_str}")

    print(f"\n{sep}\n")


# ── Step 5: Save outputs ─────────────────────────────────────────
def save_outputs(model, topics, parties, chunk_ids, texts):
    """Save topic_info CSV, chunk_topics JSON, and the trained model."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5a — topic_info.csv
    topic_info = model.get_topic_info()
    csv_path = OUTPUT_DIR / "topic_info.csv"
    topic_info.to_csv(csv_path, index=False)
    print(f"Saved topic info       → {csv_path}")

    # 5b — chunk_topics.json
    doc_records = []
    for i, (cid, party, text, tid) in enumerate(
            zip(chunk_ids, parties, texts, topics)):
        # Get keywords for this topic
        if tid == -1:
            kw = []
        else:
            kw = [w for w, _ in model.get_topic(tid)[:10]]

        doc_records.append({
            "chunk_id": cid,
            "party": party,
            "text": text,
            "topic_id": int(tid),
            "topic_keywords": kw,
        })

    ct_path = OUTPUT_DIR / "chunk_topics.json"
    with open(ct_path, "w", encoding="utf-8") as f:
        json.dump(doc_records, f, ensure_ascii=False, indent=2)
    print(f"Saved chunk topics     → {ct_path}")

    # 5c — trained model
    model_path = OUTPUT_DIR / "bertopic_model"
    model.save(str(model_path), serialization="safetensors",
               save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL_NAME)
    print(f"Saved BERTopic model   → {model_path}")


# ── Step 6: Visualisations ────────────────────────────────────────
def save_visualisations(model):
    """Generate and save interactive HTML visualisations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 6a — Intertopic distance map
    try:
        fig_topics = model.visualize_topics()
        topics_path = OUTPUT_DIR / "topics_visualization.html"
        fig_topics.write_html(str(topics_path))
        print(f"Saved topic viz        → {topics_path}")
    except Exception as e:
        print(f"⚠ Could not generate topics_visualization.html: {e}")

    # 6b — Bar chart of top keywords per topic
    try:
        fig_bar = model.visualize_barchart(top_n_topics=TARGET_TOPICS, n_words=10)
        bar_path = OUTPUT_DIR / "topic_barchart.html"
        fig_bar.write_html(str(bar_path))
        print(f"Saved barchart viz     → {bar_path}")
    except Exception as e:
        print(f"⚠ Could not generate topic_barchart.html: {e}")


# ── Summary ───────────────────────────────────────────────────────
def print_summary(model, topics):
    """Print final summary statistics."""
    unique = sorted(set(topics))
    n_topics = len(unique) - (1 if -1 in unique else 0)
    n_outliers = topics.count(-1)
    topic_info = model.get_topic_info()

    sep = "=" * 70
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"Total topics found:      {n_topics}")
    print(f"Outlier chunks (topic -1): {n_outliers}  "
          f"({n_outliers/len(topics)*100:.1f}%)")

    # Top 5 largest topics (excluding -1)
    print(f"\nTop 5 largest topics:")
    non_outlier = topic_info[topic_info["Topic"] != -1].head(5)
    for _, row in non_outlier.iterrows():
        tid = row["Topic"]
        cnt = row["Count"]
        kw = [w for w, _ in model.get_topic(tid)[:8]]
        print(f"  Topic {tid:3d} ({cnt:3d} chunks): {', '.join(kw)}")

    print(sep)


# ── Main ──────────────────────────────────────────────────────────
def main():
    texts, parties, chunk_ids, records = load_chunks(INPUT_FILE)
    model, topics, probs = run_bertopic(texts)
    print_topic_report(model, topics, parties)
    save_outputs(model, topics, parties, chunk_ids, texts)
    save_visualisations(model)
    print_summary(model, topics)


if __name__ == "__main__":
    main()
