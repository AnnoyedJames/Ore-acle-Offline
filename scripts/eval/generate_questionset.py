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
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on the path so backend imports work when running from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from backend.database.local_stores import ChromaStore
from backend.embeddings.api_generator import ApiEmbeddingGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration[]
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

_chroma_store: Optional[ChromaStore] = None
_embedder: Optional[ApiEmbeddingGenerator] = None
_page_images: Dict[str, List[Dict]] = {}    # page_title → [{url, local_filename, alt_text}]


def _get_chroma_store() -> ChromaStore:
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore(embedding_model="baai/bge-m3")
    return _chroma_store


def _get_embedder() -> ApiEmbeddingGenerator:
    global _embedder
    if _embedder is None:
        _embedder = ApiEmbeddingGenerator(model_id="baai/bge-m3")
    return _embedder


def _build_page_images() -> Dict[str, List[Dict]]:
    """Build page_title → downloaded images lookup.

    Join strategy:
      - metadata.json stores every image a page requested, each with a pre-computed
        local_filename (the .webp basename the scraper would write).
      - image_metadata.json records every file that was *actually* downloaded.
      - We match on local_filename — no URL normalisation needed, no thumbnail ambiguity.
      - metadata.json is 771 MB, so we stream it with ijson instead of json.load().

    Returns {page_title: [{url, local_filename, alt_text}, ...]}
    where url is the canonical URL from image_metadata.json.
    """
    import ijson

    # Step 1: build local_filename → canonical_url from image_metadata.json (32 MB, fast)
    img_meta_path = Path("data/processed/image_metadata.json")
    if not img_meta_path.exists():
        logger.warning("image_metadata.json not found — page images index will be empty")
        return {}
    with open(img_meta_path, encoding="utf-8") as f:
        idata = json.load(f)
    images_list = idata.get("images", idata) if isinstance(idata, dict) else idata
    downloaded: Dict[str, str] = {}  # local_filename → canonical original_url
    for img in images_list:
        fn = img.get("local_filename", "")
        url = img.get("original_url", "")
        if fn and url:
            downloaded[fn] = url
    logger.info(f"Downloaded image set: {len(downloaded)} files")

    # Step 2: stream metadata.json page-by-page to build the lookup
    metadata_path = Path("data/processed/metadata.json")
    if not metadata_path.exists():
        logger.warning("metadata.json not found — page images index will be empty")
        return {}

    page_imgs: Dict[str, List[Dict]] = {}
    logger.info("Streaming metadata.json to build page images index (may take ~10s)...")
    with open(metadata_path, "rb") as f:
        for page in ijson.items(f, "pages.item"):
            title = page.get("title", "")
            if not title:
                continue
            imgs: List[Dict] = []
            seen_fns: set = set()
            for img in page.get("images", []):
                fn = img.get("local_filename", "")
                url = img.get("url", "")
                if not fn or not url:
                    continue
                if url.lower().endswith((".svg", ".gif")):
                    continue
                if fn in seen_fns or fn not in downloaded:
                    continue
                seen_fns.add(fn)
                imgs.append({
                    "url": downloaded[fn],      # canonical URL, not the thumbnail
                    "local_filename": fn,
                    "alt_text": img.get("alt_text", ""),
                })
            if imgs:
                page_imgs[title] = imgs

    logger.info(f"Page images index: {len(page_imgs)} pages with downloaded images")
    return page_imgs


def retrieve_image_candidates(
    qa_text: str,
    source_page: str,
    top_k: int = 30,
    max_candidates: int = 30,
) -> List[Dict[str, str]]:
    """Retrieve image candidates for a Q/A pair.

    Strategy:
      1. Always include images from source_page first (guaranteed relevance).
      2. Run semantic search to find the top-K related page_titles.
      3. Add downloaded images from those pages until max_candidates is reached.

    Returns up to *max_candidates* dicts with keys: url, local_filename, alt_text.
    """
    candidates: List[Dict[str, str]] = []
    seen_filenames: set = set()

    def _add_page(title: str) -> None:
        for img in _page_images.get(title, []):
            fname = img["local_filename"]
            if fname not in seen_filenames:
                seen_filenames.add(fname)
                candidates.append(img)

    # 1. Source page images first
    _add_page(source_page)

    # 2. Semantically similar pages
    try:
        query_vec = _get_embedder().embed_query(qa_text)
        results = _get_chroma_store().query(query_vec, n_results=50)
        seen_titles: set = {source_page}
        for chunk in results:
            if len(candidates) >= max_candidates:
                break
            title = chunk.get("page_title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                _add_page(title)
    except Exception as e:
        logger.warning(f"Semantic image search failed: {e}")

    return candidates[:max_candidates]


def select_images_with_llm(
    question: str,
    answer: str,
    candidates: List[Dict[str, str]],
    model: str,
) -> List[Dict[str, str]]:
    """Ask the LLM to pick the best images from *candidates* for this Q/A pair.

    Returns a list of dicts with keys: url, local_filename.
    """
    if not candidates:
        return []

    lines = []
    for i, img in enumerate(candidates, 1):
        alt = img.get("alt_text", "")
        fname = img["local_filename"]
        lines.append(f"[Image {i}] {fname}" + (f" — {alt}" if alt else ""))
    candidates_text = "\n".join(lines)

    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Available images:\n{candidates_text}\n\n"
        "Select 3–5 images that best illustrate this answer. "
        "Prefer images that show the subject itself, relevant crafting/UI screens, or in-game screenshots. "
        "If fewer than 3 images fit well, include the closest ones anyway to reach 3. "
        "Return ONLY a JSON array of selected labels, e.g. [\"Image 1\", \"Image 3\", \"Image 5\"]."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return []
        selected_labels: list = json.loads(match.group(0))
    except Exception as e:
        logger.warning(f"Image selection LLM call failed: {e}")
        return []

    label_to_img = {f"Image {i}": img for i, img in enumerate(candidates, 1)}
    selected = []
    for label in (selected_labels if isinstance(selected_labels, list) else []):
        img = label_to_img.get(str(label))
        if img:
            selected.append({"url": img["url"], "local_filename": img["local_filename"]})
    return selected

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
You are building an evaluation dataset for a Minecraft Wiki RAG system. Generate exactly 3 questions per page — they must sound like something a real player actually typed into a Discord server, Reddit post, or Google search.

Tone benchmark (aim for these):
  "can skeletons freeze to death?"
  "whats the best y level for diamonds now"
  "do i lose xp when i die in hardcore"
  "why wont my villager restock"
  "does looting affect xp drops"
  "can you put silk touch and fortune on the same pickaxe"

NEVER write questions like these:
  "What entity state transition occurs when..." (too clinical)
  "At what Y-coordinate range does..." (jargon)
  "How does the X mechanic work exactly?" ("mechanic" is a writing crutch — say what the thing IS)
  "What happens if I [contrived edge case]?" (test-question format, not how players think)
  "What is the weird trivia about..." (players don't ask for "trivia")
  Questions with two sub-questions joined by "and" ("how does X work and can it Y?")

Rules:
1. Casual voice — lowercase ok, grammar slips ok, first-person "i" ok.
2. Answers grounded entirely in the page text. No outside knowledge.
3. Exactly one of each difficulty:
   - 'easy'  → single direct fact a new player wants. 1–2 sentence answer with a concrete detail.
   - 'medium' → needs combining 2+ facts (e.g. condition + outcome, exception, platform difference). 2–3 sentence answer.
   - 'hard'  → something a 500-hour player would still Google. Real curiosity, not a contrived edge case. Should be about a specific number, threshold, or system interaction they noticed in-game. 3+ sentence answer explaining the why/how.
4. Hard questions must NOT use the words: mechanic, conversion, reinforcement, terminal velocity, trivia, entity state, coordinate range, threshold (write around them naturally).
5. Pick up to 3 Outgoing Link URLs relevant to each question.
6. If the page has a crafting recipe, at least one question must ask how to craft or make the item.

Do not ask meta-questions about the wiki itself."""


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


def build_user_prompt(page: dict, interlinks: Dict[str, List[str]]) -> str:
    page_title = page.get("title", "")
    page_url = page.get("url", f"https://minecraft.wiki/w/{page_title.replace(' ', '_')}")
    full_text = assemble_page_text(page)

    outgoing = interlinks.get(page_title, [])
    ranked_outgoing = [
        f"https://minecraft.wiki/w/{name.replace(' ', '_')}"
        for name in outgoing[:50]
    ]

    return f"""\
Page: {page_title}
Primary URL: {page_url}
Outgoing Links: {json.dumps(ranked_outgoing)}

--- FULL WIKI PAGE TEXT START ---
{full_text}
--- FULL WIKI PAGE TEXT END ---

Generate your 3 QA pairs."""


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


def generate_qa_pairs(page: dict, interlinks: Dict[str, List[str]], model: str) -> List[dict]:
    page_title = page.get("title", "")
    timestamp = datetime.now(timezone.utc).isoformat()

    # Pass 1: Generate Q/A pairs from page text only (no images)
    text_prompt = build_user_prompt(page, interlinks)
    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text_prompt},
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
        logger.debug(f"Structured output failed (falling back to JSON parse): {e}")

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
            logger.error(f"LLM call failed for page '{page_title}': {e}")
            return []

    if not parsed_data or "items" not in parsed_data:
        logger.warning(f"No valid QA pairs returned for page '{page_title}'.")
        return []

    results = []
    for item in parsed_data["items"]:
        links = item.get("relevant_links", [])
        safe_links = links[:3] if isinstance(links, list) else []
        question = item.get("question", "")
        answer = item.get("answer", "")

        # Pass 2: semantic image retrieval + LLM selection
        relevant_images: List[Dict[str, str]] = []
        try:
            candidates = retrieve_image_candidates(f"{question} {answer}", source_page=page_title)
            if candidates:
                relevant_images = select_images_with_llm(question, answer, candidates, model)
                logger.info(f"  Image selection: {len(candidates)} candidates → {len(relevant_images)} selected")
        except Exception as e:
            logger.warning(f"Image retrieval/selection failed for '{question[:60]}': {e}")

        results.append({
            "question": question,
            "answer": answer,
            "difficulty": item.get("difficulty", "medium"),
            "relevant_links": safe_links,
            "relevant_images": relevant_images,
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
    except Exception as e:
        logger.error(f"Failed loading data: {e}")
        return

    # Build image indexes before generation so every page has candidates ready
    logger.info("Building image indexes...")
    global _page_images
    _page_images = _build_page_images()

    # Pre-warm the ChromaDB connection so the first page doesn't incur startup lag
    _get_chroma_store()

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

        pairs = generate_qa_pairs(page, interlinks, model=args.model)
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

