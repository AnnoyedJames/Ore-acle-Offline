"""
Evaluation Runner for Ore-acle Offline Hybrid RAG Pipeline.

Loads the golden question set, runs the queries through the local HybridSearch,
computes deterministic metrics (Recall@K, MRR, image hash hits), and queries
different LLM sizes for final answers. Then outputs a cleanly formatted side-by-side
Markdown report for manual review.

Models to evaluate:
-Qwen S	qwen/qwen3.5-9b	                                $0.05	$0.15
-Qwen M	qwen/qwen3.5-35b-a3b	                        TBD	TBD
-Qwen L	qwen/qwen3.5-122b-a10b	                        TBD	TBD
-GPT	openai/gpt-5.1	                                $1.25	$10.00
-Gemini Flash	google/gemini-3.1-flash-lite-preview	$0.25	$1.50
-Gemini Pro	google/gemini-3.1-pro-preview	            $2.00	$12.00

Usage:
    python scripts/eval/run_eval.py
"""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.retrieval.search import HybridSearch
from backend.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# The models you want to evaluate via OpenRouter
# (Adjust IDs if exact matches 0.6B/112B are added/changed on OpenRouter)
MODELS_TO_EVALUATE = [
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwen-2.5-72b-instruct",
]

def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset {path} not found. Run generate_questionset.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_answer(client, model_id, query, retrieved_docs):
    context = ""
    for i, doc in enumerate(retrieved_docs[:5]):
        context += f"--- Document {i+1} ---\nTitle: {doc.page_title}\nSource: {doc.page_url}\nText:\n{doc.text}\n\n"
        
    prompt = f"Using the provided documents, answer the following question about Minecraft:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600
        )
        return response.choices[0].message.content.strip(), response.usage.total_tokens
    except Exception as e:
        logger.error(f"Error querying {model_id}: {e}")
        return f"[Error: {e}]", 0

def compute_recall_mrr(retrieved_results, expected_links):
    """Compute deterministic retrieval metrics."""
    hits = [0] * len(retrieved_results)
    first_hit_rank = -1
    
    # We check if the expected link is inside the page_url of the result
    for rank, res in enumerate(retrieved_results):
        for link in expected_links:
            # simple string match for link (e.g. checking Wiki URL ending)
            expected_wiki_path = link.split("/w/")[-1] if "/w/" in link else link
            actual_wiki_path = res.page_url.split("/w/")[-1] if "/w/" in res.page_url else res.page_url
            if expected_wiki_path.lower() in actual_wiki_path.lower():
                hits[rank] = 1
                if first_hit_rank == -1:
                    first_hit_rank = rank + 1
                break
    
    recall_at_5 = 1 if sum(hits[:5]) > 0 else 0
    recall_at_10 = 1 if sum(hits[:10]) > 0 else 0
    mrr = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0
    
    return recall_at_5, recall_at_10, mrr

def build_markdown_report(metrics_log, models, output_path):
    md = ["# Offline RAG Reranking & Qwen Evaluation Report\n"]
    
    for q_idx, item in enumerate(metrics_log):
        md.append(f"## Q{q_idx+1}: {item['question']}")
        md.append(f"**Difficulty:** {item['difficulty'].upper()} | **Expected Source:** {item['source_page']}")
        md.append(f"**Expected Links:** {', '.join(item['expected_links'])}")
        md.append("")
        
        md.append("### Retrieval Performance (Deterministic)")
        md.append(f"- **Recall@5**: {item['retrieval']['recall_5']} | **Recall@10**: {item['retrieval']['recall_10']}")
        md.append(f"- **MRR**: {item['retrieval']['mrr']:.3f} | **Image Hits**: {item['retrieval'].get('image_recall', 'N/A')}\n")
        
        md.append("### Side-by-Side Model Answers")
        md.append("| Ground Truth Model (Golden) | " + " | ".join([m.split('/')[-1] for m in models]) + " |")
        md.append("|---" * (len(models) + 1) + "|")
        
        row = [item['golden_answer'].replace("\n", " ")]
        for m in models:
            row.append(item['model_answers'].get(m, "N/A").replace("\n", " "))
            
        md.append("| " + " | ".join(row) + " |")
        md.append("\n---\n")
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"Markdown report generated: {output_path}")

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("Missing OPENROUTER_API_KEY")
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    
    dataset_path = Path("data/eval/questionset.json")
    if not dataset_path.exists():
        logger.warning("No questionset found. Run generate_questionset.py first.")
        # Create a mock one if empty just for testing the script execution
        return
        
    data = load_dataset(dataset_path)
    questions = data.get("items", [])
    
    if not questions:
        logger.warning("Empty dataset.")
        return
        
    # Take up to 30 items for manual eval size
    questions = questions[:30]
    
    search_engine = HybridSearch()
    
    metrics_log = []
    
    for q in tqdm(questions, desc="Evaluating Queries"):
        item_log = {
            "question": q["question"],
            "difficulty": q.get("difficulty", "unknown"),
            "source_page": q.get("source_page", "unknown"),
            "expected_links": q.get("relevant_links", []),
            "golden_answer": q["answer"],
            "retrieval": {},
            "model_answers": {}
        }
        
        # 1. Base Retrieval
        start_ts = time.time()
        results = search_engine.search(q["question"])
        item_log["retrieval"]["latency"] = time.time() - start_ts
        
        # 2. Compute Deterministic Metrics (Text/Links)
        r5, r10, mrr = compute_recall_mrr(results, item_log["expected_links"])
        item_log["retrieval"]["recall_5"] = r5
        item_log["retrieval"]["recall_10"] = r10
        item_log["retrieval"]["mrr"] = mrr
        
        # Determine image hit rate
        expected_images = q.get("relevant_images", [])
        item_log["expected_images"] = expected_images
        actual_images = [img.get("local_filename") for res in results[:10] for img in res.images if img.get("local_filename")]
        hits = 0
        for exp in expected_images:
            if any(exp in act for act in actual_images):
                hits += 1
        item_log["retrieval"]["image_recall"] = f"{hits}/{len(expected_images)}" if expected_images else "N/A"
        
        # 3. Model Generations
        for model in MODELS_TO_EVALUATE:
            answer, tokens = generate_answer(client, model, q["question"], results)
            item_log["model_answers"][model] = answer
            
        metrics_log.append(item_log)
        
    # Output to File
    out_dir = Path("data/eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ts = int(time.time())
    json_path = out_dir / f"eval_run_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)
        
    md_path = out_dir / f"eval_report_{ts}.md"
    build_markdown_report(metrics_log, MODELS_TO_EVALUATE, md_path)
    
if __name__ == "__main__":
    main()