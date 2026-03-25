"""
Evaluation Dataset Generator for Ore-acle Offline
==================================================

Generates a gold-standard evaluation dataset by sampling chunks from the
processed Minecraft Wiki corpus and using an LLM to produce grounded
question-answer pairs.

Design Rationale
----------------
Traditional RAG evaluation requires a test set of (question, gold_answer,
gold_source_urls) triples. Manually authoring these is expensive and doesn't
scale. Instead, we exploit the fact that we *already have* the source text:

1. **Sample** a random page/chunk from the corpus.
2. **Generate** 1-2 natural questions whose answers are fully contained in
   that chunk's text.
3. **Extract** the most relevant inter-wiki links *from that chunk's own
   related_pages field* to serve as secondary gold retrieval targets.
4. The chunk's own `page_url` is always the primary gold URL.

This gives us a scalable, automatically-generated ground truth where:
- **Retrieval gold** = the source page URL + related page URLs
- **Answer gold**    = the LLM-generated reference answer (grounded in text)

The dataset is saved incrementally to `data/eval/eval_dataset.json` so
partial runs are resumable.

Output Schema (per item)
------------------------
```json
{
  "question":        "What Y-level should you mine for diamonds?",
  "gold_answer":     "Diamond ore generates between Y -64 and Y 16...",
  "gold_urls":       ["https://minecraft.wiki/w/Diamond", ...],
  "source_chunk_id": "1aa5db556551eba9",
  "source_page":     "Diamond",
  "source_text":     "<verbatim chunk text used to generate this pair>",
  "generator_model":  "google/gemini-2.5-flash-preview:thinking",
  "generated_at":    "2026-03-25T14:30:00Z"
}
```

Usage
-----
```bash
# Generate 50 QA pairs (samples 30 chunks, ~1-2 pairs each)
python scripts/eval/generate_dataset.py --num-chunks 30

# Append more pairs to an existing dataset
python scripts/eval/generate_dataset.py --num-chunks 20 --append

# Use a different model
python scripts/eval/generate_dataset.py --model google/gemini-2.5-pro-preview
```

Environment Variables
---------------------
- OPENROUTER_API_KEY : Required. Your OpenRouter API key.

Dependencies
------------
- openai >= 1.0  (OpenAI SDK, pointed at OpenRouter base URL)
- pydantic >= 2.0
- python-dotenv
"""

import argparse
import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

# OpenRouter acts as a universal LLM gateway. We use the standard OpenAI SDK
# pointed at their base URL so we can swap models with a single string change.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.warning(
        "OPENROUTER_API_KEY not set. Set it in your .env or environment "
        "before running this script."
    )

client = OpenAI(
    api_key=OPENROUTER_API_KEY or "DUMMY_KEY",
    base_url="https://openrouter.ai/api/v1",
)

# Default model: Gemini 2.5 Flash (with thinking / extended reasoning).
# Thinking models produce higher-quality, well-reasoned QA pairs because they
# can internally verify that the question is actually answerable from the text.
DEFAULT_MODEL = "google/gemini-2.5-flash-preview:thinking"

# Minimum word count for a chunk to be considered "eval-worthy".
# Short stubs don't provide enough context for meaningful questions.
MIN_CHUNK_WORDS = 50

# Wiki metadata noise that leaks through the text cleaner in some chunks.
NOISE_MARKERS = ["NewPP limit report", "Parsed by mediawiki"]


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------
class QAPair(BaseModel):
    """A single question-answer pair with supporting source URLs."""
    model_config = ConfigDict(strict=False)

    question: str
    gold_answer: str
    gold_urls: List[str]


class GeneratedEval(BaseModel):
    """Wrapper for the LLM's structured response containing 1-2 QA pairs."""
    model_config = ConfigDict(strict=False)

    items: List[QAPair]


# ---------------------------------------------------------------------------
# Chunk sampling
# ---------------------------------------------------------------------------
def load_and_filter_chunks(file_path: str) -> list:
    """Load the full chunk corpus and filter to evaluation-quality content.

    Filtering rules:
    - Skip disambiguation pages (low information density).
    - Skip chunks with fewer than MIN_CHUNK_WORDS words.
    - Skip chunks containing MediaWiki parser metadata noise.

    Returns the full filtered list (not yet sampled). Sampling is done
    separately so the caller can control the sample size.
    """
    logger.info(f"Loading chunks from {file_path} ...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data):,} total chunks.")

    valid = []
    for chunk in data:
        if chunk.get("page_type") == "disambiguation":
            continue
        text = chunk.get("text", "")
        if len(text.split()) < MIN_CHUNK_WORDS:
            continue
        if any(marker in text for marker in NOISE_MARKERS):
            continue
        valid.append(chunk)

    logger.info(f"Filtered to {len(valid):,} evaluation-eligible chunks.")
    return valid


def sample_chunks(chunks: list, count: int, seed: Optional[int] = None) -> list:
    """Randomly sample `count` chunks. Optionally set a seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
    return random.sample(chunks, min(count, len(chunks)))


# ---------------------------------------------------------------------------
# QA generation via LLM
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert evaluation dataset creator for a Minecraft Wiki RAG system.

Your job: Given a chunk of wiki text, generate 1-2 high-quality question-answer pairs
that are **strictly grounded** in the provided text.

Rules:
- Questions must sound like real user queries (natural language, not robotic).
- Answers must be fully supported by the chunk text — do NOT add outside knowledge.
- Focus on: game mechanics, crafting recipes, mob behavior, block properties, version
  history, biome details, redstone mechanics, or any concrete factual content.
- Skip if the chunk is too vague or lacks concrete facts.
- Include the source page URL as a gold URL. If the text references other wiki pages
  that are directly relevant to answering the question, include those URLs too."""


def build_user_prompt(chunk: dict) -> str:
    """Construct the user-facing prompt with the chunk text and metadata."""
    text = chunk.get("text", "")
    page_title = chunk.get("page_title", "")
    page_url = chunk.get("page_url", "")
    related = chunk.get("related_pages", [])

    # Provide related page URLs so the model can reference them as gold URLs
    # when the question spans multiple topics.
    related_urls = [
        f"https://minecraft.wiki/w/{name.replace(' ', '_')}"
        for name in related[:10]  # Cap at 10 to avoid prompt bloat
    ]

    return f"""\
Page: {page_title}
Primary URL: {page_url}
Related pages (use if relevant to the question): {json.dumps(related_urls)}

--- CHUNK TEXT START ---
{text}
--- CHUNK TEXT END ---

Generate 1-2 QA pairs as JSON matching this schema:
{{"items": [{{"question": "...", "gold_answer": "...", "gold_urls": ["..."]}}]}}"""


def parse_json_from_text(text: str) -> Optional[dict]:
    """Fallback JSON parser that extracts the first JSON object from free text.

    Some models (especially via OpenRouter) may wrap JSON in markdown fences
    or add preamble text. This function robustly extracts the JSON payload.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON inside markdown code fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any top-level JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def generate_qa_pairs(chunk: dict, model: str) -> List[dict]:
    """Call the LLM to generate QA pairs for a single chunk.

    Uses OpenRouter's structured output when possible, with a robust fallback
    to manual JSON parsing for models that don't support response_format.

    Returns a list of dicts ready for the eval dataset, with metadata attached.
    """
    user_prompt = build_user_prompt(chunk)
    timestamp = datetime.now(timezone.utc).isoformat()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    parsed_data = None

    # Strategy 1: Try structured output (works with many OpenRouter models)
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=GeneratedEval,
            temperature=0.3,
        )
        parsed_obj = response.choices[0].message.parsed
        if parsed_obj:
            parsed_data = parsed_obj.model_dump()
    except Exception as e:
        logger.debug(f"Structured output failed (expected for some models): {e}")

    # Strategy 2: Fallback to plain completion + manual JSON extraction
    if parsed_data is None:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
            )
            raw_text = response.choices[0].message.content or ""
            parsed_data = parse_json_from_text(raw_text)
        except Exception as e:
            logger.error(
                f"LLM call failed for chunk {chunk.get('chunk_id')}: {e}"
            )
            return []

    if not parsed_data or "items" not in parsed_data:
        logger.warning(
            f"No valid QA pairs returned for chunk {chunk.get('chunk_id')}."
        )
        return []

    # Attach provenance metadata to each generated pair
    results = []
    for item in parsed_data["items"]:
        results.append({
            "question": item.get("question", ""),
            "gold_answer": item.get("gold_answer", ""),
            "gold_urls": item.get("gold_urls", [chunk.get("page_url", "")]),
            "source_chunk_id": chunk.get("chunk_id", "unknown"),
            "source_page": chunk.get("page_title", ""),
            "source_text": chunk.get("text", ""),
            "generator_model": model,
            "generated_at": timestamp,
        })
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate a gold-standard eval dataset for Ore-acle RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--num-chunks", type=int, default=30,
        help="Number of chunks to sample and generate QA pairs from (default: 30).",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"OpenRouter model ID for generation (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--chunks-path", type=str, default="data/processed/chunks.json",
        help="Path to the processed chunks JSON file.",
    )
    parser.add_argument(
        "--output", type=str, default="data/eval/eval_dataset.json",
        help="Path to write the eval dataset JSON.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to an existing dataset file instead of overwriting.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    # Validate input
    if not Path(args.chunks_path).exists():
        logger.error(f"Chunks file not found: {args.chunks_path}")
        return

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load and sample
    all_chunks = load_and_filter_chunks(args.chunks_path)
    sampled = sample_chunks(all_chunks, args.num_chunks, seed=args.seed)
    logger.info(f"Sampled {len(sampled)} chunks for QA generation.")

    # Load existing dataset if appending
    dataset: list = []
    if args.append and Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"Loaded existing dataset with {len(dataset)} items (append mode).")

    # Generate QA pairs
    new_count = 0
    for i, chunk in enumerate(sampled):
        page = chunk.get("page_title", "???")
        logger.info(f"[{i+1}/{len(sampled)}] Generating QA for: {page}")

        pairs = generate_qa_pairs(chunk, model=args.model)
        if pairs:
            dataset.extend(pairs)
            new_count += len(pairs)
            logger.info(f"  -> {len(pairs)} pair(s) generated.")

            # Save incrementally after each successful generation so partial
            # runs are never lost (important for long runs with API rate limits).
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        else:
            logger.warning(f"  -> No pairs generated (skipped).")

    # Final summary
    logger.info("=" * 60)
    logger.info(f"Dataset generation complete.")
    logger.info(f"  New pairs generated : {new_count}")
    logger.info(f"  Total dataset size  : {len(dataset)}")
    logger.info(f"  Output file         : {args.output}")
    logger.info(f"  Model used          : {args.model}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()