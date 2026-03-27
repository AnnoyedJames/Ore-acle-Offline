"""
Evaluation Question Set Generator for Ore-acle Offline
======================================================

Generates a gold-standard evaluation question set by targeting the most important
(high PageRank) whole pages from the Minecraft Wiki corpus and using an LLM to 
produce grounded questions, answers, difficulty levels, and relevant links.

Design Rationale
----------------
Traditional RAG evaluation requires a test set of triples. To avoid "chunk leakage"
and evaluate true long-context retrieval logic, we feed the LLM the ENTIRE text 
of a target wiki page (e.g. "Creeper", "Crafting") determined by a mathematically
robust PageRank-based popularity heuristic.

1. **Score** all wiki pages using PageRank + Content Density modifiers.
2. **Select** the top N most important pages.
3. **Generate** 3-5 natural questions whose answers are contained in the text.
4. **Extract** relevant inter-wiki links based on valid outgoing links.

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
  "generator_model": "google/gemini-3.1-flash-lite-preview",
  "generated_at":    "2026-03-25T14:30:00Z"
}
```
"""

import argparse
import base64
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

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

DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------
class QAPair(BaseModel):
    """A single question-answer pair with supporting metadata."""
    model_config = ConfigDict(strict=False)

    question: str
    answer: str
    relevant_links: List[str]
    relevant_images: List[str] = []
    difficulty: str


class GeneratedEval(BaseModel):
    """Wrapper for the LLM's structured response containing QA pairs."""
    model_config = ConfigDict(strict=False)

    items: List[QAPair]


# ---------------------------------------------------------------------------
# PageRank & Scoring Logic
# ---------------------------------------------------------------------------
def compute_pagerank(
    interlinks: Dict[str, List[str]], 
    damping_factor: float = 0.85, 
    max_iterations: int = 50, 
    tol: float = 1e-6
) -> Dict[str, float]:
    nodes = set(interlinks.keys())
    for targets in interlinks.values():
        nodes.update(targets)
    
    N = len(nodes)
    if N == 0:
        return {}

    pr = {node: 1.0 / N for node in nodes}
    out_degree = {node: len(targets) for node, targets in interlinks.items()}
    
    incoming = defaultdict(list)
    for src, targets in interlinks.items():
        for target in targets:
            incoming[target].append(src)
            
    dangling_nodes = [n for n in nodes if out_degree.get(n, 0) == 0]

    for i in range(max_iterations):
        new_pr = {}
        dangling_sum = sum(pr[n] for n in dangling_nodes)
        base_pr = (1.0 - damping_factor) / N + damping_factor * (dangling_sum / N)
        
        diff = 0.0
        for node in nodes:
            sum_in = sum(pr[inc] / out_degree[inc] for inc in incoming[node])
            new_pr[node] = base_pr + damping_factor * sum_in
            diff += abs(new_pr[node] - pr[node])
            
        pr = new_pr
        if diff < tol:
            logger.info(f"PageRank converged at iteration {i+1}")
            break
            
    return pr


def select_top_pages(metadata: List[dict], interlinks: Dict[str, List[str]], top_k: int) -> List[dict]:
    logger.info("Computing PageRank scores...")
    pr_scores = compute_pagerank(interlinks)
    
    word_counts = {p['title']: p.get('word_count', 0) for p in metadata}
    scored_pages = []
    
    for page in metadata:
        title = page['title']
        pr = pr_scores.get(title, 0.0)
        wc = word_counts.get(title, 0)
        
        # 1. Log-scale
        pr_factor = math.log10((pr * 1000000) + 1)
        wc_factor = math.log10(max(wc, 1))
        
        # 2. Content Density Mutliplier
        has_infobox = 1 if page.get('infobox') else 0
        img_count = len(page.get('images', []))
        content_multiplier = 1.0 + (has_infobox * 0.5) + (min(img_count, 50) * 0.01)
        
        # 3. Penalize Meta-Pages
        is_meta = bool(re.search(r'(?i)edition\b|\b1\.\d+|\b\d{2}w\d+[a-z]?|launcher|mojang|protocol|tracker', title))
        penalty = 0.3 if is_meta else 1.0
        
        final_score = pr_factor * wc_factor * content_multiplier * penalty
        scored_pages.append((final_score, page))
        
    scored_pages.sort(key=lambda x: x[0], reverse=True)
    return [p[1] for p in scored_pages[:top_k]]


# ---------------------------------------------------------------------------
# QA generation via LLM
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert evaluation dataset creator for a Minecraft Wiki RAG system.

Your job: Given the ENTIRE text of a highly popular Minecraft Wiki page, generate 3-5 high-quality testing questions that are **strictly grounded** in the provided text.

Rules:
- Questions must sound like real user queries (natural language, not robotic).
- Answers must be fully supported by the page text  do NOT add outside knowledge.
- Determine the 'difficulty' of the question: 'easy' (direct fact extraction), 'medium' (requires combining 2+ facts across the page), or 'hard' (complex reasoning/edge cases).
- Look at the provided "Outgoing Links" array. For each question, identify up to 3 links from that array that are highly relevant to the specific question topic to serve as gold-standard retrieval targets. Return them as full URLs (e.g. "https://minecraft.wiki/w/Diamond").
- If relevant images are provided in the user prompt array visually, you may identify their "image_hash" (up to 2) that are highly relevant to the question. Return them in the "relevant_images" list. If no image is relevant, leave it empty.
- Focus on: game mechanics, crafting recipes, mob behavior, block properties, etc.
- Do not ask meta-questions about the wiki itself."""


def assemble_page_text(page: dict) -> str:
    sections = page.get("sections", [])
    if not isinstance(sections, list):
        return ""
        
    full_text = []
    for sec in sections:
        level = sec.get("level", 2)
        heading = sec.get("heading", "")
        text = sec.get("text", "")
        
        if heading:
            full_text.append(f"{'#' * level} {heading}\n")
        if text:
            full_text.append(f"{text}\n")
            
    return "\n".join(full_text)


def build_user_prompt(page: dict, interlinks: Dict[str, List[str]], images_to_show: List[dict]) -> str:
    page_title = page.get("title", "")
    page_url = page.get("url", f"https://minecraft.wiki/w/{page_title.replace(' ', '_')}")
    full_text = assemble_page_text(page)
    
    outgoing = interlinks.get(page_title, [])
    # Cap outgoing links to 50 so we don't blow up the prompt entirely
    ranked_outgoing = [
        f"https://minecraft.wiki/w/{name.replace(' ', '_')}"
        for name in outgoing[:50]
    ]

    return f"""\
Page: {page_title}
Primary URL: {page_url}
Outgoing Links context: {json.dumps(ranked_outgoing)}
Images context (hashes): {json.dumps([img['image_hash'] for img in images_to_show])}

--- FULL WIKI PAGE TEXT START ---
{full_text}
--- FULL WIKI PAGE TEXT END ---

Generate 3-5 QA pairs as JSON matching this schema:
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


def generate_qa_pairs(page: dict, interlinks: Dict[str, List[str]], image_metadata: Dict[str, list], model: str) -> List[dict]:
    # Find matching images
    page_filename = Path(page.get("file_path", "")).name
    images_for_page = image_metadata.get(page_filename, [])[:10]  # Cap at 10 to avoid payload explosion
    
    user_prompt = build_user_prompt(page, interlinks, images_for_page)
    timestamp = datetime.now(timezone.utc).isoformat()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    user_content = [{"type": "text", "text": user_prompt}]
    
    # Add images
    for img in images_for_page:
        try:
            with open(img["file_path"], "rb") as image_file:
                b64_img = base64.b64encode(image_file.read()).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{b64_img}"}
                })
        except Exception as e:
            logger.warning(f"Could not load image {img['file_path']}: {e}")
    
    messages.append({"role": "user", "content": user_content})

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
                f"LLM call failed for page {page.get('title')}: {e}"
            )
            return []

    if not parsed_data or "items" not in parsed_data:
        logger.warning(
            f"No valid QA pairs returned for page {page.get('title')}."
        )
        return []

    results = []
    for item in parsed_data["items"]:
        links = item.get("relevant_links", [])
        safe_links = links[:3] if isinstance(links, list) else []
        
        results.append({
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "difficulty": item.get("difficulty", "medium"),
            "relevant_links": safe_links,
            "relevant_images": item.get("relevant_images", []),
            "source_page": page.get("title", ""),
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
        "--num-pages", type=int, default=10,
        help="Number of top pages to evaluate and generate QA pairs for (default: 10).",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"OpenRouter model ID for generation (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output", type=str, default="data/eval/questionset.json",
        help="Path to write the eval questionset JSON.",
    )
    args = parser.parse_args()

    # Load Data
    logger.info("Loading metadata & interlinks...")
    try:
        with open('data/processed/metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)["pages"]
            
        with open('data/processed/interlinks.json', 'r', encoding='utf-8') as f:
            interlinks = json.load(f)["graph"]
            
        with open('data/processed/image_metadata.json', 'r', encoding='utf-8') as f:
            raw_image_data = json.load(f)["images"]
            
        image_metadata = defaultdict(list)
        for img in raw_image_data:
            for sp in img.get("source_pages", []):
                image_metadata[sp].append(img)
    except Exception as e:
        logger.error(f"Failed loading data: {e}")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Get the highest quality pages
    top_pages = select_top_pages(metadata, interlinks, args.num_pages)
    logger.info(f"Selected {len(top_pages)} top priority pages for generation.")

    dataset: list = []
    if Path(args.output).exists():
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            logger.info(f"Loaded existing dataset with {len(dataset)} items (will append).")
        except json.JSONDecodeError:
            logger.warning("Existing dataset is invalid JSON. Starting fresh.")

    # Only process pages we haven't already processed in the existing dataset
    processed_titles = {item.get("source_page") for item in dataset}

    new_count = 0
    for i, page in enumerate(top_pages):
        page_title = page.get("title", '???')
        
        if page_title in processed_titles:
            logger.info(f"[{i+1}/{len(top_pages)}] Skipping '{page_title}' (already in dataset).")
            continue
            
        logger.info(f"[{i+1}/{len(top_pages)}] Generating QA for: {page_title} (Words: {page.get('word_count')})")

        pairs = generate_qa_pairs(page, interlinks, image_metadata, model=args.model)
        if pairs:
            dataset.extend(pairs)
            new_count += len(pairs)
            logger.info(f"  -> {len(pairs)} pair(s) generated.")

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        else:
            logger.warning(f"  -> No pairs generated for {page_title}.")

    logger.info("=" * 60)
    logger.info(f"Dataset generation complete.")
    logger.info(f"  New questions generated : {new_count}")
    logger.info(f"  Total questionset size  : {len(dataset)}")
    logger.info(f"  Output file             : {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

