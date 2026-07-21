#!/usr/bin/env python3
"""
inspect_and_clean.py

Reads each manifesto from manifestos_extracted/, detects noise patterns
programmatically, prints a noise report per file, cleans the file, and
saves the result to manifestos_cleaned/{party}_clean.txt.

All cleaning decisions are derived from the data itself — no hardcoded
party-specific rules.

Devanagari unicode characters (U+0900–U+097F) are treated as valid
alphanumeric for the alphanumeric-ratio check (relevant for sp_raw.txt).
"""

import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_DIR = Path("manifestos_extracted")
OUTPUT_DIR = Path("manifestos_cleaned")
REPEAT_THRESHOLD = 5        # lines appearing more than this many times → noise
ALNUM_RATIO_MIN = 0.40      # lines where < 40 % chars are "alphanumeric" → noise

# Files to process (basename → party slug used for output naming)
FILES = {
    "bjp_raw.txt":    "bjp",
    "inc_raw.txt":    "inc",
    "cpim_raw.txt":   "cpim",
    "tmc_raw.txt":    "tmc",
    "ncpsp_raw.txt":  "ncpsp",
    "dmk_raw.txt":    "dmk",
    "sp_raw.txt":     "sp",
    "ncp_ap_raw.txt": "ncp_ap",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

# Devanagari block: U+0900 – U+097F
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# Patterns
_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_EMAIL_RE = re.compile(r'\S+@\S+\.\S+')


def _is_alnum_or_devanagari(ch: str) -> bool:
    """Return True if the character is alphanumeric (ASCII or any script)
    or belongs to the Devanagari block."""
    if ch.isalnum():
        return True
    # Devanagari dependent vowels, virama, nukta etc. are not flagged as
    # "alphanumeric" by Python but are legitimate text characters.
    if '\u0900' <= ch <= '\u097F':
        return True
    return False


def alnum_ratio(line: str) -> float:
    """Fraction of characters in *line* that are alphanumeric (incl. Devanagari)."""
    chars = [ch for ch in line if not ch.isspace()]
    if not chars:
        return 0.0
    good = sum(1 for ch in chars if _is_alnum_or_devanagari(ch))
    return good / len(chars)


def is_standalone_digit_line(line: str) -> bool:
    """True if the stripped line is composed entirely of digits (page numbers etc.)."""
    s = line.strip()
    return bool(s) and s.isdigit()


def is_only_symbols(line: str) -> bool:
    """True if the line contains no alphanumeric chars at all (only symbols/punctuation/whitespace)."""
    s = line.strip()
    if not s:
        return False  # blank lines handled separately
    return not any(_is_alnum_or_devanagari(ch) for ch in s)




def has_url(line: str) -> bool:
    return bool(_URL_RE.search(line))


def has_email(line: str) -> bool:
    return bool(_EMAIL_RE.search(line))


# ─── Per-file analysis ────────────────────────────────────────────────────────

def analyse_file(lines: list[str]):
    """
    Return a dict mapping each noise category to a list of (line_number, line_text) tuples,
    plus the set of repeated-line strings.
    """
    noise = {
        "repeated_lines":   [],
        "low_alnum":        [],
        "standalone_digit": [],
        "urls":             [],
        "emails":           [],
        "only_symbols":     [],
    }

    # ── Step 1: find repeated lines (stripped, non-empty) ─────────────────
    counter = Counter()
    for line in lines:
        s = line.strip()
        if s:
            counter[s] += 1

    repeated_strings = {s for s, c in counter.items() if c > REPEAT_THRESHOLD}

    # ── Step 2: classify each line ────────────────────────────────────────
    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()

        # Skip blank lines (we keep them; collapsing happens later)
        if not stripped:
            continue

        # Check categories (a line can match multiple, but we record each)
        if stripped in repeated_strings:
            noise["repeated_lines"].append((idx, stripped))

        if is_standalone_digit_line(stripped):
            noise["standalone_digit"].append((idx, stripped))
            continue  # no need to check further for a pure digit line

        if is_only_symbols(stripped):
            noise["only_symbols"].append((idx, stripped))
            continue

        if has_url(stripped):
            noise["urls"].append((idx, stripped))
            # URL lines might still be valid; we flag but don't skip others

        if has_email(stripped):
            noise["emails"].append((idx, stripped))

        if alnum_ratio(stripped) < ALNUM_RATIO_MIN:
            noise["low_alnum"].append((idx, stripped))

    return noise, repeated_strings


# ─── Noise report ─────────────────────────────────────────────────────────────

def print_noise_report(filename: str, noise: dict, repeated_strings: set):
    """Pretty-print the noise report for one file."""
    print(f"\n{'='*80}")
    print(f"  NOISE REPORT: {filename}")
    print(f"{'='*80}")

    categories = [
        ("repeated_lines",   "Repeated lines (>{} occurrences)".format(REPEAT_THRESHOLD)),
        ("low_alnum",        "Low alphanumeric ratio (<{:.0%})".format(ALNUM_RATIO_MIN)),
        ("standalone_digit", "Standalone digit lines"),
        ("urls",             "Lines containing URLs"),
        ("emails",           "Lines containing emails"),
        ("only_symbols",     "Symbol/punctuation-only lines"),
    ]

    for key, label in categories:
        items = noise[key]
        print(f"\n  ▸ {label}: {len(items)} lines detected")
        if items:
            # Show up to 10 unique examples
            seen = set()
            shown = 0
            for lineno, text in items:
                if text not in seen and shown < 10:
                    preview = text[:90] + ("…" if len(text) > 90 else "")
                    print(f"      L{lineno:>5d}: {preview}")
                    seen.add(text)
                    shown += 1
            if len(items) > shown:
                print(f"      ... and {len(items) - shown} more")

    if repeated_strings:
        print(f"\n  ▸ Distinct repeated strings: {len(repeated_strings)}")
        for s in sorted(repeated_strings)[:15]:
            preview = s[:80] + ("…" if len(s) > 80 else "")
            print(f"      • {preview}")
        if len(repeated_strings) > 15:
            print(f"      ... and {len(repeated_strings) - 15} more")


# ─── Cleaning ──────────────────────────────────────────────────────────────────

def clean_lines(lines: list[str], noise: dict, repeated_strings: set) -> list[str]:
    """
    Remove noisy lines and collapse excessive blank lines.
    Returns the cleaned list of lines.
    """
    # Build a set of line numbers to remove
    remove_linenos: set[int] = set()

    for key in noise:
        for lineno, _ in noise[key]:
            remove_linenos.add(lineno)

    cleaned = []
    prev_blank = False

    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()

        # Remove flagged noisy lines
        if idx in remove_linenos:
            continue

        # Remove form-feed characters (PDF page breaks)
        if stripped == '\f' or stripped == '':
            if stripped == '\f':
                continue
            # Collapse consecutive blank lines into at most one
            if prev_blank:
                continue
            cleaned.append("")
            prev_blank = True
            continue

        # Clean form-feed characters embedded in text
        cleaned_line = raw.replace('\f', '').rstrip()
        cleaned.append(cleaned_line)
        prev_blank = False

    # Strip leading/trailing blank lines
    while cleaned and cleaned[0].strip() == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1].strip() == "":
        cleaned.pop()

    return cleaned


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for filename, party in FILES.items():
        filepath = INPUT_DIR / filename
        if not filepath.exists():
            print(f"[SKIP] {filepath} not found")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

        original_count = len(raw_lines)

        # Analyse
        noise, repeated_strings = analyse_file(raw_lines)

        # Report
        print_noise_report(filename, noise, repeated_strings)

        # Clean
        cleaned = clean_lines(raw_lines, noise, repeated_strings)
        cleaned_count = len(cleaned)

        # Save
        out_path = OUTPUT_DIR / f"{party}_clean.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned) + "\n")

        summary_rows.append((filename, party, original_count, cleaned_count,
                             original_count - cleaned_count))

        print(f"\n  ✓ Saved → {out_path}  ({cleaned_count} lines)")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'File':<25s} {'Original':>10s} {'Cleaned':>10s} {'Removed':>10s} {'% Removed':>10s}")
    print(f"  {'-'*65}")
    total_orig = total_clean = total_rem = 0
    for filename, party, orig, clean, rem in summary_rows:
        pct = (rem / orig * 100) if orig else 0
        print(f"  {filename:<25s} {orig:>10d} {clean:>10d} {rem:>10d} {pct:>9.1f}%")
        total_orig += orig
        total_clean += clean
        total_rem += rem
    pct_total = (total_rem / total_orig * 100) if total_orig else 0
    print(f"  {'-'*65}")
    print(f"  {'TOTAL':<25s} {total_orig:>10d} {total_clean:>10d} {total_rem:>10d} {pct_total:>9.1f}%")
    print()


if __name__ == "__main__":
    main()
