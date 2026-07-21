#!/usr/bin/env python3
"""
chunk.py — Manifesto chunking pipeline
Reads cleaned manifesto files, applies per-file chunking strategies,
and produces semantically coherent chunks of 60-200 words.
"""

import json
import re
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────
INPUT_DIR = Path("manifestos_cleaned")
OUTPUT_FILE = INPUT_DIR / "all_chunks.json"
MIN_WORDS = 60
MAX_WORDS = 200

# ── File configurations ──────────────────────────────────────────
CONFIGS = {
    "bjp_clean.txt":    {"party": "BJP",   "group": "fragmented", "skip": 0},
    "inc_clean.txt":    {"party": "INC",   "group": "medium",     "skip": 10},
    "cpim_clean.txt":   {"party": "CPIM",  "group": "medium",     "skip": 0},
    "tmc_clean.txt":    {"party": "TMC",   "group": "dense",      "skip": 0},
    "ncpsp_clean.txt":  {"party": "NCPSP", "group": "fragmented", "skip": 0},
    "dmk_clean.txt":    {"party": "DMK",   "group": "dense",      "skip": 0},
    "sp_clean.txt":     {"party": "SP",    "group": "fragmented", "skip": 0},
    "ncp_ap_clean.txt": {"party": "NCPAP", "group": "fragmented", "skip": 0},
}

# ── Noise patterns to strip ──────────────────────────────────────
NOISE_PATTERNS = {
    "NCPSP": [
        re.compile(r'^\d*\s*\|?\s*Nationalist Congress Party'),
        re.compile(r'Nationalist Congress Party.*\|\s*\d+$'),
    ],
    "NCPAP": [
        re.compile(r'OKSABHA|LOKSABHA.*ELECTION', re.IGNORECASE),
    ],
    "SP": [
        re.compile(r'॥{2,}\s*\d'),
    ],
}


def is_noise_line(line, party):
    """Check if a line matches known noise patterns."""
    for pat in NOISE_PATTERNS.get(party, []):
        if pat.search(line):
            return True
    return False


# ── Heading detection ─────────────────────────────────────────────

def _is_all_caps(s, min_alpha=8):
    """Check if string is ALL CAPS with enough alpha characters."""
    alpha = re.sub(r'[^A-Za-z]', '', s)
    return len(alpha) >= min_alpha and s == s.upper() and bool(re.search(r'[A-Z]', s))


def _is_title_case(words, min_ratio=0.5):
    """Check if enough words start with uppercase."""
    alpha_words = [w for w in words if w[0].isalpha()]
    if not alpha_words:
        return False
    cap = sum(1 for w in alpha_words if w[0].isupper())
    return cap >= len(alpha_words) * min_ratio


def is_heading(line, party, prev_blank, next_blank):
    """Determine if a line is a section heading."""
    s = line.strip()
    if not s or not prev_blank:
        return False

    words = s.split()
    wc = len(words)

    if party == "BJP":
        if wc > 8:
            return False
        if _is_all_caps(s):
            return True
        if wc <= 6 and next_blank and not s.endswith(('.', ',')):
            return _is_title_case(words)
        return False

    elif party == "INC":
        if wc > 8:
            return False
        if _is_all_caps(s, min_alpha=5):
            return True
        if wc <= 4 and next_blank and not s.endswith(('.', ',')):
            return _is_title_case(words)
        return False

    elif party == "CPIM":
        if wc < 2 or wc > 8:
            return False
        if not next_blank or s.endswith(('.', ',')):
            return False
        return _is_title_case(words)

    elif party == "TMC":
        if _is_all_caps(s, min_alpha=5) and wc <= 6:
            return True
        if s.startswith("What the ") and s.endswith(":"):
            return True
        if s == "Key Goal":
            return True
        return False

    elif party == "NCPSP":
        if wc < 1 or wc > 6:
            return False
        if not next_blank or s.endswith(('.', ',')):
            return False
        if not re.search(r'[A-Za-z]', s):
            return False
        return _is_title_case(words)

    elif party == "DMK":
        if s.endswith(':') and wc <= 8:
            return True
        if wc <= 4 and next_blank and not s.endswith(('.', ',')):
            return _is_title_case(words)
        return False

    elif party == "SP":
        if s.startswith(('>',  '»', '-')):
            return False
        if wc > 6:
            return False
        if not re.search(r'[\u0900-\u097F]', s):
            return False
        if s.endswith('।'):
            return False
        return next_blank

    elif party == "NCPAP":
        if wc < 1 or wc > 6:
            return False
        if s.endswith(('.', ',')):
            return False
        if not re.search(r'[A-Za-z]', s):
            return False
        return _is_title_case(words, min_ratio=0.4)

    return False


# ── Sentence splitting ────────────────────────────────────────────

def _force_split_long(text, max_w=MAX_WORDS):
    """
    Force-split text that exceeds max_w words on clause boundaries.
    Tries to find a balanced split near the midpoint so neither half
    is too small (< MIN_WORDS).
    """
    total = wc(text)
    if total <= max_w:
        return [text]

    words = text.split()
    target_mid = total // 2  # Aim for midpoint

    # Find all candidate split points (after commas, semicolons, », danda)
    split_chars = {',', ';', '»', '।'}
    candidates = []
    word_idx = 0
    for i, ch in enumerate(text):
        if ch == ' ':
            word_idx += 1
        if ch in split_chars and MIN_WORDS <= word_idx <= total - MIN_WORDS:
            candidates.append((i, word_idx))

    if candidates:
        # Pick the candidate closest to the midpoint
        best_pos, best_wid = min(candidates, key=lambda x: abs(x[1] - target_mid))
        # Split after the delimiter character + any whitespace
        left = text[:best_pos + 1].strip()
        right = text[best_pos + 1:].strip()
        if left and right:
            result = []
            # Recursively split if halves are still too long
            result.extend(_force_split_long(left, max_w))
            result.extend(_force_split_long(right, max_w))
            return result

    # Last resort: hard split at word boundary near max_w
    result = []
    for i in range(0, len(words), max_w):
        chunk = " ".join(words[i:i + max_w])
        if chunk:
            result.append(chunk)
    return result


def split_sentences(text, party):
    """Split text into sentences using appropriate boundary markers."""
    if party == "SP":
        # Split on Hindi danda (।)
        parts = re.split(r'(?<=।)\s*', text)
    else:
        # Split on . ! ? followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', text)
    result = [p.strip() for p in parts if p.strip()]
    if not result:
        result = [text]
    # Force-split any individual sentence that's still too long
    final = []
    for s in result:
        final.extend(_force_split_long(s))
    return final


def wc(text):
    """Count words in text."""
    return len(text.split())


# ── Section splitting ─────────────────────────────────────────────

def split_into_sections(lines, party):
    """
    Split lines into (heading, content_text) sections.
    Each section is a tuple of (heading_string, body_text).
    """
    sections = []
    cur_heading = ""
    cur_lines = []

    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        if is_noise_line(stripped, party):
            continue

        prev_blank = (i == 0) or (lines[i - 1].strip() == "")
        next_blank = (i >= len(lines) - 1) or (lines[i + 1].strip() == "")

        if is_heading(stripped, party, prev_blank, next_blank):
            # Save the accumulated section
            if cur_lines or cur_heading:
                sections.append((cur_heading, " ".join(cur_lines)))
            cur_heading = stripped
            cur_lines = []
        else:
            cur_lines.append(stripped)

    # Flush final section
    if cur_lines or cur_heading:
        sections.append((cur_heading, " ".join(cur_lines)))

    return sections


# ── Chunking strategies ───────────────────────────────────────────

def _build_chunks_from_sentences(sentences, max_w=MAX_WORDS):
    """
    Greedily build chunks by accumulating sentences up to max_w words.
    Returns a list of chunk strings.
    """
    chunks = []
    buf = []
    buf_wc = 0

    for sent in sentences:
        swc = wc(sent)
        if buf and buf_wc + swc > max_w:
            chunks.append(" ".join(buf))
            buf = [sent]
            buf_wc = swc
        else:
            buf.append(sent)
            buf_wc += swc

    if buf:
        chunks.append(" ".join(buf))

    # Post-process: force-split any chunk still over max_w
    final = []
    for c in chunks:
        if wc(c) > max_w:
            final.extend(_force_split_long(c, max_w))
        else:
            final.append(c)
    return final


def chunk_dense(sections, party):
    """
    Dense files (DMK, TMC): split on headings, then sliding-window
    of max 200 words splitting only on sentence boundaries.
    """
    all_chunks = []

    for heading, body in sections:
        if not body.strip():
            continue

        body_wc = wc(body)

        # Small enough to keep as one chunk
        if body_wc <= MAX_WORDS:
            text = f"{heading}\n{body}".strip() if heading else body.strip()
            all_chunks.append(text)
            continue

        # Split into sentences, then build chunks
        sents = split_sentences(body, party)
        raw_chunks = _build_chunks_from_sentences(sents, MAX_WORDS)

        for idx, chunk in enumerate(raw_chunks):
            # Prepend heading to the first chunk of each section
            if idx == 0 and heading:
                chunk = f"{heading}\n{chunk}"
            all_chunks.append(chunk.strip())

    return all_chunks


def chunk_medium(sections, party):
    """
    Medium files (INC, CPIM): group consecutive paragraphs under
    same heading until hitting ~150 words, split on sentence boundaries.
    """
    target = 150
    all_chunks = []

    for heading, body in sections:
        if not body.strip():
            continue

        body_wc = wc(body)

        if body_wc <= MAX_WORDS:
            text = f"{heading}\n{body}".strip() if heading else body.strip()
            all_chunks.append(text)
            continue

        # Split on sentence boundaries and build chunks targeting ~150 words
        sents = split_sentences(body, party)
        raw_chunks = _build_chunks_from_sentences(sents, target)

        for idx, chunk in enumerate(raw_chunks):
            if idx == 0 and heading:
                chunk = f"{heading}\n{chunk}"
            all_chunks.append(chunk.strip())

    return all_chunks


def chunk_fragmented(sections, party):
    """
    Fragmented files (BJP, NCPSP, SP, NCPAP): merge consecutive
    short paragraphs under same heading until reaching 80-150 words.
    """
    target_min = 80
    target_max = 150
    all_chunks = []

    for heading, body in sections:
        if not body.strip():
            continue

        body_wc = wc(body)

        # If already in the sweet spot, keep as one chunk
        if body_wc <= target_max:
            text = f"{heading}\n{body}".strip() if heading else body.strip()
            all_chunks.append(text)
            continue

        # Split into sentences and build chunks targeting 80-150 words
        sents = split_sentences(body, party)
        raw_chunks = _build_chunks_from_sentences(sents, target_max)

        for idx, chunk in enumerate(raw_chunks):
            if idx == 0 and heading:
                chunk = f"{heading}\n{chunk}"
            all_chunks.append(chunk.strip())

    return all_chunks


# ── Merge undersized chunks ──────────────────────────────────────

def merge_small_chunks(chunks, min_w=MIN_WORDS, max_ceiling=None):
    """
    Merge chunks that are below min_w words with adjacent chunks.
    Runs two passes: forward merge then backward merge for stragglers.
    """
    if not chunks:
        return chunks

    merge_ceiling = max_ceiling if max_ceiling is not None else MAX_WORDS

    # Pass 1: forward merge — small chunk absorbs into previous
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        prev_wc = wc(merged[-1])
        cur_wc = wc(chunk)

        if prev_wc < min_w and prev_wc + cur_wc <= merge_ceiling:
            merged[-1] = merged[-1] + " " + chunk
        elif cur_wc < min_w and prev_wc + cur_wc <= merge_ceiling:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    # Pass 2: backward merge — any remaining small chunk merges with next
    result = []
    i = 0
    while i < len(merged):
        cur = merged[i]
        cur_wc_val = wc(cur)
        if cur_wc_val < min_w and i + 1 < len(merged):
            nxt_wc = wc(merged[i + 1])
            if cur_wc_val + nxt_wc <= merge_ceiling:
                result.append(cur + " " + merged[i + 1])
                i += 2
                continue
        result.append(cur)
        i += 1

    # Handle trailing tiny chunk
    if len(result) > 1 and wc(result[-1]) < min_w:
        if wc(result[-2]) + wc(result[-1]) <= merge_ceiling:
            result[-2] = result[-2] + " " + result[-1]
            result.pop()

    return result


# ── Main pipeline ─────────────────────────────────────────────────

STRATEGY_MAP = {
    "dense":      chunk_dense,
    "medium":     chunk_medium,
    "fragmented": chunk_fragmented,
}


def process_file(filename, config):
    """Process a single manifesto file and return its chunks."""
    party = config["party"]
    group = config["group"]
    skip = config["skip"]

    filepath = INPUT_DIR / filename
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header lines (e.g., INC table of contents)
    if skip > 0:
        lines = lines[skip:]

    # Split into sections
    sections = split_into_sections(lines, party)

    # Apply chunking strategy
    strategy = STRATEGY_MAP[group]
    chunks = strategy(sections, party)

    # Merge undersized chunks
    chunks = merge_small_chunks(chunks)

    # Clean up whitespace
    chunks = [re.sub(r'\s+', ' ', c).strip() for c in chunks]
    chunks = [c for c in chunks if c]  # Remove empty

    # ── Final enforcement pass ────────────────────────────────────
    # Force-split any chunk still over MAX_WORDS
    enforced = []
    for c in chunks:
        if wc(c) > MAX_WORDS:
            enforced.extend(_force_split_long(c, MAX_WORDS))
        else:
            enforced.append(c)
    chunks = enforced

    # Force-merge any chunk still under MIN_WORDS with neighbor
    # Use relaxed ceiling so force-split fragments can be reabsorbed
    chunks = merge_small_chunks(chunks, min_w=MIN_WORDS, max_ceiling=MAX_WORDS + 15)

    return chunks


def main():
    all_chunks = []

    print("=" * 70)
    print("MANIFESTO CHUNKING PIPELINE")
    print("=" * 70)

    for filename, config in CONFIGS.items():
        party = config["party"]
        chunks = process_file(filename, config)

        # Build output records
        for i, text in enumerate(chunks):
            all_chunks.append({
                "party": party,
                "chunk_id": f"{party}_{i}",
                "text": text,
            })

        # Per-file stats
        word_counts = [wc(c) for c in chunks]
        if word_counts:
            print(f"\n{party:6s} ({config['group']:11s}) | "
                  f"chunks: {len(chunks):4d} | "
                  f"min: {min(word_counts):3d} | "
                  f"max: {max(word_counts):3d} | "
                  f"avg: {sum(word_counts)/len(word_counts):5.1f}")
        else:
            print(f"\n{party:6s} | NO CHUNKS PRODUCED — check input file")

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print(f"TOTAL CHUNKS: {len(all_chunks)}")
    print(f"OUTPUT: {OUTPUT_FILE}")

    total_wc = [wc(c["text"]) for c in all_chunks]
    print(f"OVERALL — min: {min(total_wc)}, max: {max(total_wc)}, "
          f"avg: {sum(total_wc)/len(total_wc):.1f}")

    # Word count distribution
    under_60 = sum(1 for w in total_wc if w < 60)
    in_range = sum(1 for w in total_wc if 60 <= w <= 200)
    over_200 = sum(1 for w in total_wc if w > 200)
    print(f"DISTRIBUTION — <60w: {under_60}, 60-200w: {in_range}, >200w: {over_200}")
    print("=" * 70)


if __name__ == "__main__":
    main()
