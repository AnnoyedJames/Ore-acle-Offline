"""
Questionset Heuristic Auditor

Runs three automated quality checks against the golden eval dataset:

  1. Answer length vs difficulty  — hard/medium answers that are too short
  2. Missing relevant_links       — questions with no retrieval targets
  3. Near-duplicate questions     — cosine similarity on TF-IDF vectors
  4. Title-verbatim questions     — question contains the page title verbatim

Prints a summary report and writes flagged items to data/eval/audit_flags.json.

Usage:
    python scripts/eval/audit_questionset.py
    python scripts/eval/audit_questionset.py --dataset data/eval/questionset.json
    python scripts/eval/audit_questionset.py --dup-threshold 0.85
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATASET_DEFAULT = Path("data/eval/questionset.json")
FLAGS_DEFAULT = Path("data/eval/audit_flags.json")

# Minimum word counts per difficulty level
MIN_ANSWER_WORDS = {"easy": 8, "medium": 20, "hard": 40}

# Default cosine-similarity threshold for duplicate detection
DEFAULT_DUP_THRESHOLD = 0.82


# ---------------------------------------------------------------------------
# TF-IDF helpers (no sklearn dependency required — pure numpy)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z]+", text.lower())


def _build_tfidf(docs: list[list[str]]) -> np.ndarray:
    """Build a TF-IDF matrix (n_docs × vocab) using pure Python + numpy."""
    # Vocabulary
    vocab: dict[str, int] = {}
    for tokens in docs:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    V = len(vocab)
    N = len(docs)

    # TF matrix
    tf = np.zeros((N, V), dtype=np.float32)
    for i, tokens in enumerate(docs):
        for t in tokens:
            tf[i, vocab[t]] += 1
        if tokens:
            tf[i] /= len(tokens)

    # IDF
    df = np.count_nonzero(tf, axis=0).astype(np.float32)
    idf = np.log((N + 1) / (df + 1)) + 1.0  # smoothed

    tfidf = tf * idf

    # L2-normalise each row
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tfidf / norms


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_answer_length(items: list[dict]) -> list[dict]:
    """Flag answers that are too short for their declared difficulty."""
    flags = []
    for i, item in enumerate(items):
        diff = item.get("difficulty", "easy")
        threshold = MIN_ANSWER_WORDS.get(diff, 8)
        word_count = len(item.get("answer", "").split())
        if word_count < threshold:
            flags.append({
                "check": "short_answer",
                "index": i,
                "source_page": item.get("source_page"),
                "difficulty": diff,
                "answer_words": word_count,
                "threshold": threshold,
                "question": item.get("question"),
                "answer": item.get("answer"),
            })
    return flags


def check_missing_links(items: list[dict]) -> list[dict]:
    """Flag questions that have no relevant_links."""
    flags = []
    for i, item in enumerate(items):
        if not item.get("relevant_links"):
            flags.append({
                "check": "missing_links",
                "index": i,
                "source_page": item.get("source_page"),
                "difficulty": item.get("difficulty"),
                "question": item.get("question"),
            })
    return flags


def check_near_duplicates(items: list[dict], threshold: float) -> list[dict]:
    """Flag pairs of questions with cosine similarity above threshold."""
    questions = [item.get("question", "") for item in items]
    tokenized = [_tokenize(q) for q in questions]
    # Skip docs that are completely empty
    non_empty_idx = [i for i, t in enumerate(tokenized) if t]
    sub_tokens = [tokenized[i] for i in non_empty_idx]

    if len(sub_tokens) < 2:
        return []

    tfidf = _build_tfidf(sub_tokens)

    flags = []
    # Use batched dot product; memory-safe for ~1500 vectors at float32
    scores = tfidf @ tfidf.T  # (n × n)

    for row in range(len(non_empty_idx)):
        for col in range(row + 1, len(non_empty_idx)):
            sim = float(scores[row, col])
            if sim >= threshold:
                i, j = non_empty_idx[row], non_empty_idx[col]
                flags.append({
                    "check": "near_duplicate",
                    "similarity": round(sim, 4),
                    "index_a": i,
                    "index_b": j,
                    "source_page_a": items[i].get("source_page"),
                    "source_page_b": items[j].get("source_page"),
                    "question_a": items[i].get("question"),
                    "question_b": items[j].get("question"),
                })
    return flags


def check_title_verbatim(items: list[dict]) -> list[dict]:
    """Flag questions that contain the page title verbatim (too obvious)."""
    flags = []
    for i, item in enumerate(items):
        title = item.get("source_page", "").lower()
        question = item.get("question", "").lower()
        if title and title in question:
            flags.append({
                "check": "title_verbatim",
                "index": i,
                "source_page": item.get("source_page"),
                "difficulty": item.get("difficulty"),
                "question": item.get("question"),
            })
    return flags


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(items: list[dict], all_flags: list[dict], threshold: float) -> None:
    by_check: dict[str, list] = {}
    for f in all_flags:
        by_check.setdefault(f["check"], []).append(f)

    total = len(items)
    print(f"\n{'='*60}")
    print(f" Questionset Audit — {total} pairs")
    print(f"{'='*60}")

    # Short answers
    sa = by_check.get("short_answer", [])
    print(f"\n[1] Short answers:  {len(sa)} flagged  ({len(sa)/total*100:.1f}%)")
    per_diff: dict[str, int] = {}
    for f in sa:
        per_diff[f["difficulty"]] = per_diff.get(f["difficulty"], 0) + 1
    for d, c in sorted(per_diff.items()):
        print(f"    {d:8s}: {c}")
    if sa:
        worst = sorted(sa, key=lambda x: x["answer_words"])[:3]
        print("  Shortest examples:")
        for f in worst:
            print(f"    [{f['difficulty']:6s} {f['answer_words']:2d}w] [{f['source_page']}] {f['question']!r}")
            print(f"           → {f['answer']!r}")

    # Missing links
    ml = by_check.get("missing_links", [])
    print(f"\n[2] Missing links:  {len(ml)} flagged  ({len(ml)/total*100:.1f}%)")
    if ml[:3]:
        for f in ml[:3]:
            print(f"    [{f['source_page']}] {f['question']!r}")

    # Near duplicates
    nd = by_check.get("near_duplicate", [])
    print(f"\n[3] Near-duplicates (≥{threshold}):  {len(nd)} pairs flagged")
    if nd[:5]:
        for f in sorted(nd, key=lambda x: -x["similarity"])[:5]:
            print(f"    sim={f['similarity']:.3f}  [{f['source_page_a']}] {f['question_a']!r}")
            print(f"           ↔   [{f['source_page_b']}] {f['question_b']!r}")

    # Title verbatim
    tv = by_check.get("title_verbatim", [])
    print(f"\n[4] Title verbatim: {len(tv)} flagged  ({len(tv)/total*100:.1f}%)")
    if tv[:3]:
        for f in tv[:3]:
            print(f"    [{f['source_page']}] {f['question']!r}")

    # Overall
    flagged_indices = {f.get("index") or f.get("index_a") for f in all_flags}
    print(f"\n{'='*60}")
    print(f" Total unique items with ≥1 flag: {len(flagged_indices)} / {total}  ({len(flagged_indices)/total*100:.1f}%)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the golden eval questionset.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--output", type=Path, default=FLAGS_DEFAULT)
    parser.add_argument("--dup-threshold", type=float, default=DEFAULT_DUP_THRESHOLD,
                        help="Cosine similarity threshold for duplicate detection (default: 0.82)")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)
    if not isinstance(items, list):
        items = items.get("items", [])
    print(f"Loaded {len(items)} items from {args.dataset}")

    print("Running checks...")
    flags: list[dict[str, Any]] = []
    flags += check_answer_length(items)
    flags += check_missing_links(items)
    print(f"  Checks 1 & 2 done. Running near-duplicate detection ({len(items)} vectors)...")
    flags += check_near_duplicates(items, args.dup_threshold)
    flags += check_title_verbatim(items)

    print_report(items, flags, args.dup_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2, ensure_ascii=False)
    print(f"Flags written to {args.output}  ({len(flags)} total flags)")


if __name__ == "__main__":
    main()
