"""
patch_gold_spans.py — Post-hoc enrichment pass for questionset.json
====================================================================

For every entry in the existing question set this script adds two new fields:

    gold_spans : list[dict]
        Verbatim sentences or short phrases extracted from the source page(s)
        that directly support the answer.  Each span carries:
          - "text"        : exact quote from the page HTML
          - "source_page" : page title the quote came from
          - "hop"         : 1-indexed hop number (1 for primary page,
                            2+ for secondary pages in multi-hop questions)

    paraphrases : list[str]
        2 re-worded variants of the question (already in schema from
        generate_questionset.py — this script fills it for legacy entries).

    multi_hop : bool
        True if the answer requires evidence from more than one page.

The script is idempotent: entries that already have all three fields are
skipped unless --force is passed.

Usage
-----
    # Dry-run first 10 entries to validate prompts / cost
    python scripts/eval/patch_gold_spans.py --limit 10 --dry-run

    # Full patch pass (resumes from last checkpoint automatically)
    python scripts/eval/patch_gold_spans.py

    # Re-patch even entries that already have gold_spans
    python scripts/eval/patch_gold_spans.py --force
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
DATASET_PATH = Path("data/eval/questionset.json")
HTML_DIR = Path("data/raw/html")

client = OpenAI(
    api_key=OPENROUTER_API_KEY or "DUMMY_KEY",
    base_url="https://openrouter.ai/api/v1",
)

# ---------------------------------------------------------------------------
# HTML → plain text
# ---------------------------------------------------------------------------
def _html_to_text(html: str) -> str:
    """Very lightweight HTML stripping — keeps sentence structure intact."""
    # Remove script/style blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Replace block-level tags with newlines
    html = re.sub(r"</(p|div|li|h[1-6]|tr|td|th)>", "\n", html, flags=re.IGNORECASE)
    # Strip remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities
    for ent, ch in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                    ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"')]:
        html = html.replace(ent, ch)
    # Collapse whitespace
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def _load_page_text(page_title: str) -> Optional[str]:
    """Load and clean the HTML for *page_title*. Returns None if not found."""
    # HTML files are stored as <title>.html with URL-encoded names
    from urllib.parse import quote
    candidates = [
        HTML_DIR / f"{page_title}.html",
        HTML_DIR / f"{quote(page_title, safe='')}.html",
        HTML_DIR / f"{page_title.replace(' ', '_')}.html",
    ]
    for path in candidates:
        if path.exists():
            raw = path.read_text(encoding="utf-8", errors="replace")
            return _html_to_text(raw)

    # Fallback: case-insensitive glob
    pattern = page_title.lower().replace(" ", "_")
    for path in HTML_DIR.glob("*.html"):
        if path.stem.lower().replace("%20", "_") == pattern:
            raw = path.read_text(encoding="utf-8", errors="replace")
            return _html_to_text(raw)

    logger.warning(f"  HTML not found for page: '{page_title}'")
    return None


# ---------------------------------------------------------------------------
# LLM prompt for gold span extraction
# ---------------------------------------------------------------------------
SPAN_SYSTEM = """\
You are building a retrieval evaluation dataset.  Your job is to extract the
exact sentences (verbatim quotes) from one or more wiki pages that together
provide the evidence needed to answer a question.

Rules:
- ONLY quote text that appears verbatim in the supplied page text(s).
- Each span should be 1–4 sentences, long enough to be self-contained but not
  an entire paragraph.
- If the answer needs evidence from multiple pages, produce spans from each
  required page. Do NOT include a page if it only provides background context
  that isn't strictly necessary to answer the question.
- Set multi_hop to true only when a reader CANNOT answer the question from
  any single page alone — both pages are necessary.
- Assign hop numbers starting at 1 for the primary (source_page) page.

Return ONLY valid JSON with this exact schema:
{
  "multi_hop": false,
  "gold_spans": [
    {"text": "<exact verbatim quote>", "source_page": "<page title>", "hop": 1},
    ...
  ]
}
"""


def _build_span_prompt(
    question: str,
    answer: str,
    pages: Dict[str, str],   # page_title → plain text
) -> str:
    sections = []
    for title, text in pages.items():
        # Truncate very long pages to avoid context limits (keep first 12 000 chars)
        excerpt = text[:12_000] + ("\n[... page truncated ...]" if len(text) > 12_000 else "")
        sections.append(f"=== PAGE: {title} ===\n{excerpt}")

    pages_block = "\n\n".join(sections)

    return (
        f"Question: {question}\n\n"
        f"Expected answer (abstractive — do NOT copy this verbatim; use it only to "
        f"guide which sentences to extract): {answer}\n\n"
        f"{pages_block}\n\n"
        "Extract the gold spans."
    )


def extract_gold_spans(
    question: str,
    answer: str,
    source_page: str,
    relevant_links: List[str],
    model: str,
    retry: int = 2,
) -> Optional[dict]:
    """Call the LLM to extract gold_spans + multi_hop flag.

    Returns {"multi_hop": bool, "gold_spans": [...]} or None on failure.
    """
    # Collect page texts (source page + any additional linked pages)
    page_titles = [source_page]
    for link in relevant_links:
        # Convert URL --> page title: https://minecraft.wiki/w/Blue_Ice --> Blue Ice
        m = re.search(r"/w/(.+)$", link)
        if m:
            t = m.group(1).replace("_", " ")
            if t not in page_titles:
                page_titles.append(t)

    pages: Dict[str, str] = {}
    for title in page_titles:
        text = _load_page_text(title)
        if text:
            pages[title] = text

    if not pages:
        logger.warning(f"  No page text available — skipping gold span extraction")
        return None

    prompt = _build_span_prompt(question, answer, pages)

    for attempt in range(1, retry + 2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SPAN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
            # Strip markdown fences if present
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            data = json.loads(raw)
            spans = data.get("gold_spans", [])
            if not isinstance(spans, list) or not spans:
                raise ValueError("Empty gold_spans list")
            # Validate structure
            for s in spans:
                if not isinstance(s.get("text"), str) or not s["text"].strip():
                    raise ValueError(f"Invalid span: {s}")
            return {
                "multi_hop": bool(data.get("multi_hop", False)),
                "gold_spans": spans,
            }
        except Exception as e:
            logger.warning(f"  Span extraction attempt {attempt} failed: {e}")
            if attempt <= retry:
                time.sleep(1)

    return None


# ---------------------------------------------------------------------------
# Paraphrase generation (mirrors generate_questionset.py)
# ---------------------------------------------------------------------------
PARAPHRASE_SYSTEM = """\
You generated search query variations of a given question.
Rules:
- Give queries that look like real, concise Google searches (keyword-focused).
- Do NOT use natural language slang like "yo", "anybody know".
- Use choppy, keyword-dense language (e.g. "minecraft water bucket recipe", "depth strider speed boost").
- Use natural language questions like how people google (e.g. "How do I get an elytra?", "Who is steve?", etc.)
- Preserve the core meaning exactly.
Return ONLY a JSON array of 2 strings, e.g. ["variant 1", "variant 2"].
"""


def generate_paraphrases(question: str, model: str) -> List[str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PARAPHRASE_SYSTEM},
                {"role": "user", "content": f"Original: {question}\n\nGenerate 2 paraphrase variants."},
            ],
            temperature=0.7,
        )
        raw = resp.choices[0].message.content or ""
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not m:
            return []
        variants = json.loads(m.group(0))
        return [v for v in variants if isinstance(v, str) and v.strip()][:2]
    except Exception as e:
        logger.warning(f"  Paraphrase generation failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc gold span + paraphrase patching.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N entries (0 = all)")
    parser.add_argument("--force", action="store_true", help="Re-patch entries that already have gold_spans")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts but make no LLM calls and don't save")
    parser.add_argument("--spans-only", action="store_true", help="Skip paraphrase generation")
    parser.add_argument("--paraphrases-only", action="store_true", help="Skip span extraction")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        dataset: List[dict] = json.load(f)

    entries = dataset[:args.limit] if args.limit else dataset
    total = len(entries)
    patched = 0
    skipped = 0

    for i, entry in enumerate(entries):
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        source_page = entry.get("source_page", "")
        relevant_links = entry.get("relevant_links", [])

        needs_spans = not args.paraphrases_only and (args.force or not entry.get("gold_spans"))
        needs_paraphrases = not args.spans_only and (args.force or not entry.get("paraphrases"))

        if not needs_spans and not needs_paraphrases:
            skipped += 1
            continue

        logger.info(f"[{i+1}/{total}] '{question[:70]}'")

        if args.dry_run:
            pages_avail = []
            for link in [source_page] + [re.sub(r".*/w/", "", l).replace("_", " ") for l in relevant_links]:
                if _load_page_text(link):
                    pages_avail.append(link)
            logger.info(f"  [DRY-RUN] pages available: {pages_avail}")
            logger.info(f"  [DRY-RUN] needs_spans={needs_spans}  needs_paraphrases={needs_paraphrases}")
            continue

        changed = False

        if needs_spans:
            result = extract_gold_spans(question, answer, source_page, relevant_links, args.model)
            if result:
                entry["gold_spans"] = result["gold_spans"]
                entry["multi_hop"] = result["multi_hop"]
                logger.info(
                    f"  gold_spans: {len(result['gold_spans'])} span(s)  "
                    f"multi_hop={result['multi_hop']}"
                )
                changed = True
            else:
                logger.warning(f"  gold_spans extraction failed — entry left unchanged")

        if needs_paraphrases:
            paraphrases = generate_paraphrases(question, args.model)
            entry["paraphrases"] = paraphrases
            logger.info(f"  paraphrases: {len(paraphrases)} variant(s)")
            changed = True

        if changed:
            patched += 1
            # Save after every entry so a crash doesn't lose progress
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

        # Polite rate-limit pause
        time.sleep(0.3)

    logger.info("=" * 60)
    logger.info(f"Patch pass complete.")
    logger.info(f"  Processed : {total}  Patched: {patched}  Skipped (already done): {skipped}")
    logger.info(f"  Output    : {path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
