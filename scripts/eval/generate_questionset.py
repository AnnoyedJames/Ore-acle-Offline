"""
Evaluation Question Set Generator for Ore-acle Offline
======================================================

Generates a gold-standard evaluation question set by sampling chunks from the
processed Minecraft Wiki corpus and using an LLM to produce grounded
questions, answers, difficulty levels, and relevant links.

Design Rationale
----------------
Traditional RAG evaluation requires a test set of (question, gold_answer,
gold_source_urls) triples. Manually authoring these is expensive and doesn't
scale. Instead, we exploit the fact that we *already have* the source text:

1. **Sample** a random page/chunk from the corpus.
2. **Generate** 1-2 natural questions whose answers are fully contained in
   that chunk's text, determining their difficulty (easy/medium/hard).
3. **Extract** up to 3 most relevant inter-wiki links *from that chunk's own
   related_pages field* to serve as secondary gold retrieval targets.

This gives us a scalable, automatically-generated ground truth where:
- **Retrieval gold** = the source page + up to 3 relevant links
- **Answer gold**    = the LLM-generated reference answer (grounded in text)

The dataset is saved incrementally to `data/eval/questionset.json` so
partial runs are resumable.

Output Schema (per item)
------------------------
```json
{
  "question":        "What Y-level should you mine for diamonds?",
  "answer":          "Diamond ore generates between Y -64 and Y 16...",
  "difficulty":      "easy",
  "relevant_links":  ["https://minecraft.wiki/w/Diamond", "https://minecraft.wiki/w/Ore"],
  "source_page":     "Diamond",
  "source_chunk_id": "1aa5db556551eba9",
  "source_text":     "<verbatim chunk text used to generate this pair>",
  "generator_model": "google/gemini-2.5-flash-preview:thinking",
  "generated_at":    "2026-03-25T14:30:00Z"
}
```

Usage
-----
```bash
# Generate 50 QA pairs (samples 30 chunks, ~1-2 pairs each)
python scripts/eval/generate_questionset.py --num-chunks 30

# Append more pairs to an existing dataset
python scripts/eval/generate_questionset.py --num-chunks 20 --append

# Use a different model
python scripts/eval/generate_questionset.py --model google/gemini-2.5-pro-preview
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

DEFAULT_MODEL = "google/gemini-2.5-flash-preview:thinking"

MIN_CHUNK_WORDS = 50
NOISE_MARKERS = ["NewPP limit report", "Parsed by mediawiki"]


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------
class QAPair(BaseModel):
    """A single question-answer pair with supporting metadata."""
    model_config = ConfigDict(strict=False)

    question: str
    answer: str
    relevant_links: List[str]
    difficulty: str


class GeneratedEval(BaseModel):
    """Wrapper for the LLM's structured response containing 1-2 QA pairs."""
    model_config = ConfigDict(strict=False)

    items: List[QAPair]


# ---------------------------------------------------------------------------
# Chunk sampling
# ---------------------------------------------------------------------------
def load_and_filter_chunks(file_path: str) -> list:
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
    if seed is not None:
        random.seed(seed)
    return random.sample(chunks, min(count, len(chunks)))


# ---------------------------------------------------------------------------
# QA generation via LLM
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert evaluation dataset creator for a Minecraft Wiki RAG system.

Your job: Given a chunk of wiki text, generate 1-2 high-quality testing questions
that are **strictly grounded** in the provided text.

Rules:
- Questions must sound like real user queries (natural language, not robotic).
- Answers must be fully supported by the chunk text  do NOT add outside knowledge.
- Determine the 'difficulty' of the question: 'easy' (direct fact extraction), 'medium' (requires combining 2+ facts), or 'hard' (complex reasoning/edge cases).
- Identify up to 3 most relevant links (from the provided related pages list or derived from the text) that someone answering this question would want to read.
- Focus on: game mechanics, crafting recipes, mob behavior, block properties, etc.
- Skip if the chunk is too vague or lacks concrete facts."""


def build_user_prompt(chunk: dict) -> str:
    text = chunk.get("text", "")
    page_title = chunk.get("page_title", "")
    page_url = chunk.get("page_url", "")
    related = chunk.get("related_pages", [])

    # Provide related page URLs so the model can reference them
    related_urls = [
        f"https://minecraft.wiki/w/{name.replace(' ', '_')}"
        for name in related[:10]  # Cap at 10 to avoid prompt bloat
    ]

    return f"""\
Page: {page_title}
Primary URL: {page_url}
Related pages context: {json.dumps(related_urls)}

--- CHUNK TEXT START ---
{text}
--- CHUNK TEXT END ---

Generate 1-2 QA pairs as JSON matching this schema:
{{"items": [{{"question": "...", "answer": "...", "relevant_links": ["url1", "url2"], "difficulty": "easy|medium|hard"}}]}}"""


def parse_json_from_text(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def generate_qa_pairs(chunk: dict, model: str) -> List[dict]:
    user_prompt = build_user_prompt(chunk)
    timestamp = datetime.now(timezone.utc).isoformat()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    parsed_data = None

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
        logger.debug(f"Structured output failed (falling back): {e}")

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

    results = []
    for item in parsed_data["items"]:
        # Ensure max 3 links are stored
        links = item.get("relevant_links", [])
        safe_links = links[:3] if isinstance(links, list) else []
        
        results.append({
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "difficulty": item.get("difficulty", "medium"),
            "relevant_links": safe_links,
            "source_page": chunk.get("page_title", ""),
            "source_chunk_id": chunk.get("chunk_id", "unknown"),
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
        description="Generate a gold-standard eval question set for Ore-acle Offline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--output", type=str, default="data/eval/questionset.json",
        help="Path to write the eval questionset JSON.",
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

    if not Path(args.chunks_path).exists():
        logger.error(f"Chunks file not found: {args.chunks_path}")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    all_chunks = load_and_filter_chunks(args.chunks_path)
    sampled = sample_chunks(all_chunks, args.num_chunks, seed=args.seed)
    logger.info(f"Sampled {len(sampled)} chunks for QA generation.")

    dataset: list = []
    if args.append and Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"Loaded existing dataset with {len(dataset)} items (append mode).")

    new_count = 0
    for i, chunk in enumerate(sampled):
        page = chunk.get("page_title", "???")
        logger.info(f"[{i+1}/{len(sampled)}] Generating QA for: {page}")

        pairs = generate_qa_pairs(chunk, model=args.model)
        if pairs:
            dataset.extend(pairs)
            new_count += len(pairs)
            logger.info(f"  -> {len(pairs)} pair(s) generated.")

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        else:
            logger.warning(f"  -> No pairs generated (skipped).")

    logger.info("=" * 60)
    logger.info(f"Dataset generation complete.")
    logger.info(f"  New questions generated : {new_count}")
    logger.info(f"  Total questionset size  : {len(dataset)}")
    logger.info(f"  Output file             : {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

