"""
Evaluation Runner for Ore-acle Offline - Two-Phase Ablation Framework.

Phase 1  RETRIEVER  (no LLM)
  Varies one axis at a time while holding others at their defaults.
    --axis embedding   -> 4 embedding models
    --axis search      -> 3 search modes (semantic / keyword / hybrid)
    --axis chunking    -> 2 chunking strategies (section_aware / langchain)
  Metrics: Recall@5, Recall@10, Precision@10, MRR

Phase 2  GENERATOR  (best retrieval config from Phase 1)
  Runs the winning retrieval pipeline, then sends retrieved chunks to
  each of the 4 LLMs.  Measures answer quality.
    --phase generator
  Metrics: Token-level F1, ROUGE-L, human spot-check (manual)

Usage:
    # Free - keyword/semantic/hybrid comparison
    python scripts/eval/run_eval.py --phase retriever --axis search

    # Needs re-embedding for each model (local, free except Gemini API)
    python scripts/eval/run_eval.py --phase retriever --axis embedding

    # Best retrieval -> LLM generation comparison
    python scripts/eval/run_eval.py --phase generator

    # Override retrieval defaults for generator phase
    python scripts/eval/run_eval.py --phase generator \
        --embedding BAAI/bge-m3 --search-mode hybrid --chunking section_aware
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env before any backend imports
from tqdm import tqdm

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config.settings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    LLM_MODELS,
    settings,
)
from backend.database.local_stores import ChromaStore, SQLiteStore
from backend.embeddings import get_embedder
from backend.retrieval.search import HybridSearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = Path("data/eval/questionset.json")
RESULTS_DIR = Path("data/eval/results")

# Defaults - these are the "held constant" values when varying one axis
DEFAULT_SEARCH_MODE = "hybrid"
DEFAULT_CHUNKING = "section_aware"

# Axes
EMBEDDING_AXIS_MODELS = list(EMBEDDING_MODELS.keys())
SEARCH_AXIS_MODES = ["semantic", "keyword", "hybrid"]
RRF_ALPHA_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9]
CHUNKING_AXIS_STRATEGIES = ["section_aware", "langchain"]

# Per-chunking-strategy metadata: (chunks_file, sqlite_db_path)
CHUNKING_META: dict[str, dict] = {
    "section_aware": {
        "chunks_file": settings.chunks_file,
        "sqlite_path": settings.sqlite_db_path,
        "collection_suffix": "",  # no suffix → uses model name alone
    },
    "langchain": {
        "chunks_file": Path("data/processed/chunks_langchain.json"),
        "sqlite_path": Path("data/sqlite_fts_langchain.db"),
        "collection_suffix": "__langchain",
    },
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path, limit: int | None = None) -> list[dict]:
    """Load the golden QA dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run generate_questionset.py first."
        )
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    # Support both flat list and {"items": [...]} wrapper
    items = data if isinstance(data, list) else data.get("items", [])
    if limit:
        items = items[:limit]
    logger.info(f"Loaded {len(items)} questions from {path}")
    return items


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _normalise_wiki_path(url: str) -> str:
    """Extract and lower-case the wiki path for comparison."""
    if "/w/" in url:
        return url.split("/w/")[-1].lower()
    return url.lower()


def compute_retrieval_metrics(
    results: list,
    expected_links: list[str],
) -> dict:
    """Compute Recall@5, Recall@10, Precision@10, MRR against expected links.

    Recall@K = fraction of expected pages that appear at least once in top-K results.
    Precision@K = fraction of top-K results that come from an expected page.
    MRR = 1 / rank of the first result from any expected page.
    """
    if not expected_links:
        return {"recall@5": 0, "recall@10": 0, "precision@10": 0, "mrr": 0.0}

    expected_paths = {_normalise_wiki_path(link) for link in expected_links}
    first_hit_rank = -1

    # Track which expected pages were covered and how many results were relevant
    found_at_5: set[str] = set()
    found_at_10: set[str] = set()
    relevant_in_10 = 0

    for rank, res in enumerate(results[:10]):
        actual_path = _normalise_wiki_path(res.page_url)
        matched = next(
            (exp for exp in expected_paths if exp in actual_path or actual_path in exp),
            None,
        )
        if matched is not None:
            relevant_in_10 += 1
            found_at_10.add(matched)
            if rank < 5:
                found_at_5.add(matched)
            if first_hit_rank == -1:
                first_hit_rank = rank + 1

    n_expected = len(expected_paths)
    return {
        "recall@5": len(found_at_5) / n_expected,
        "recall@10": len(found_at_10) / n_expected,
        "precision@10": relevant_in_10 / min(10, len(results)) if results else 0,
        "mrr": (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0,
    }


def compute_image_recall(
    results: list,
    expected_images: list[str],
) -> dict:
    """Compute image hash hit rate in the top-10 results."""
    if not expected_images:
        return {"image_hits": 0, "image_total": 0, "image_recall": 0.0}

    actual = set()
    for res in results[:10]:
        for img in res.images:
            fname = img.get("local_filename", "")
            if fname:
                actual.add(fname)

    # expected_images may be strings or dicts {url, local_filename}
    expected_fnames = [
        (exp if isinstance(exp, str) else exp.get("local_filename", ""))
        for exp in expected_images
    ]
    expected_fnames = [f for f in expected_fnames if f]

    hits = sum(1 for exp in expected_fnames if any(exp in a for a in actual))
    return {
        "image_hits": hits,
        "image_total": len(expected_images),
        "image_recall": hits / len(expected_images),
    }


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. Gemma 4)."""
    import re
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _tokenise(text: str) -> list[str]:
    """Whitespace tokenisation for F1 / ROUGE-L."""
    return text.lower().split()


def compute_token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference."""
    pred_toks = set(_tokenise(prediction))
    ref_toks = set(_tokenise(reference))
    if not pred_toks or not ref_toks:
        return 0.0
    common = pred_toks & ref_toks
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score."""
    pred_toks = _tokenise(prediction)
    ref_toks = _tokenise(reference)
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_length(pred_toks, ref_toks)
    precision = lcs / len(pred_toks)
    recall = lcs / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Build search engine for a given config
# ---------------------------------------------------------------------------
# Cache ChromaStore/SQLiteStore instances so ChromaDB is only loaded once
# ---------------------------------------------------------------------------
_chroma_cache: dict[str, ChromaStore] = {}
_sqlite_cache: dict[str, SQLiteStore] = {}


def _build_search(
    embedding_model: str,
    search_mode: str,
    chunking: str,
    rrf_alpha: float | None = None,
    rrf_k: int | None = None,
) -> HybridSearch:
    """Construct a HybridSearch wired to the right stores/embedder.

    ChromaStore and SQLiteStore are cached by key so the HNSW index is
    only loaded from disk once per process, regardless of how many configs
    are evaluated.
    """
    meta = CHUNKING_META.get(chunking, CHUNKING_META["section_aware"])
    embedder = get_embedder(embedding_model)
    collection_key = embedding_model + meta["collection_suffix"]

    if collection_key not in _chroma_cache:
        logger.info(f"Initialising ChromaDB collection '{collection_key}' (first use) ...")
        _chroma_cache[collection_key] = ChromaStore(embedding_model=collection_key)
    chroma = _chroma_cache[collection_key]

    sqlite_key = str(meta["sqlite_path"])
    if sqlite_key not in _sqlite_cache:
        _sqlite_cache[sqlite_key] = SQLiteStore(db_path=meta["sqlite_path"])
    sqlite = _sqlite_cache[sqlite_key]

    return HybridSearch(chroma=chroma, sqlite=sqlite, embedder=embedder,
                        rrf_alpha=rrf_alpha, rrf_k=rrf_k)


def _citation_faithfulness(answer: str, results: list, source_page: str) -> Optional[float]:
    """Return 1.0 if the model cited a source from the expected page, 0.0 if it
    didn't, or None if the expected page was not retrieved (unevaluable).

    Detection is heuristic: looks for ``[Source N]`` or ``[N]`` patterns in the
    answer and checks whether source N corresponds to the expected page.
    """
    import re
    # Find which 1-indexed source slots belong to the expected page
    source_page_lower = (source_page or "").lower()
    matching = {
        i + 1
        for i, res in enumerate(results[:5])
        if source_page_lower and source_page_lower in getattr(res, "page_title", "").lower()
    }
    if not matching:
        return None  # can't assess — page wasn't retrieved

    cited = {int(m) for m in re.findall(r"\[(?:Source\s+)?(\d+)\]", answer)}
    return 1.0 if matching & cited else 0.0


def _build_context_string(results: list, max_sources: int = 5) -> str:
    """Format retrieved results into a context string for the LLM."""
    parts = []
    for i, res in enumerate(results[:max_sources]):
        parts.append(
            f"[Source {i + 1}]\n"
            f"Page: {res.page_title}\n"
            f"Section: {res.section_heading}\n"
            f"Content:\n{res.text}"
        )
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Load chunks.json for text hydration
# ---------------------------------------------------------------------------

def _load_chunks_lookup(path: Path | None = None) -> dict[str, dict]:
    """Load a chunks JSON file into a {chunk_id: chunk_dict} lookup."""
    cpath = path or settings.chunks_file
    if not cpath.exists():
        logger.warning(f"{cpath} not found")
        return {}
    with open(cpath, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c for c in chunks}


# ===================================================================
# PHASE 1 - RETRIEVER EVALUATION
# ===================================================================

def run_retriever_axis(
    axis: str,
    questions: list[dict],
    chunks_lookup: dict[str, dict],
) -> dict:
    """Run one retriever axis, return aggregated metrics + per-question log."""

    if axis == "embedding":
        configs = [
            {"embedding": m, "search": DEFAULT_SEARCH_MODE, "chunking": DEFAULT_CHUNKING}
            for m in EMBEDDING_AXIS_MODELS
        ]
    elif axis == "search":
        configs = [
            {"embedding": DEFAULT_EMBEDDING_MODEL, "search": m, "chunking": DEFAULT_CHUNKING}
            for m in SEARCH_AXIS_MODES
        ]
    elif axis == "chunking":
        configs = [
            {"embedding": DEFAULT_EMBEDDING_MODEL, "search": DEFAULT_SEARCH_MODE, "chunking": m}
            for m in CHUNKING_AXIS_STRATEGIES
        ]
    elif axis == "rrf":
        # Sweep semantic weight alpha at k=20; include pure modes as baselines
        configs = [
            {"embedding": DEFAULT_EMBEDDING_MODEL, "search": "semantic",
             "chunking": DEFAULT_CHUNKING, "rrf_alpha": None, "rrf_k": None,
             "label": f"{DEFAULT_EMBEDDING_MODEL}|semantic|{DEFAULT_CHUNKING}"},
            {"embedding": DEFAULT_EMBEDDING_MODEL, "search": "keyword",
             "chunking": DEFAULT_CHUNKING, "rrf_alpha": None, "rrf_k": None,
             "label": f"{DEFAULT_EMBEDDING_MODEL}|keyword|{DEFAULT_CHUNKING}"},
        ] + [
            {"embedding": DEFAULT_EMBEDDING_MODEL, "search": "hybrid",
             "chunking": DEFAULT_CHUNKING, "rrf_alpha": a, "rrf_k": 20,
             "label": f"hybrid|α={a:.2f}|k=20"}
            for a in RRF_ALPHA_SWEEP
        ]
    else:
        raise ValueError(f"Unknown axis: {axis}")

    all_results: dict[str, list[dict]] = {}

    # Cache of per-chunking loaded lookups to avoid redundant re-loads
    _lookup_cache: dict[str, dict] = {}

    for cfg in configs:
        label = cfg.get("label", f"{cfg['embedding']}|{cfg['search']}|{cfg['chunking']}")
        logger.info(f"\n{'='*60}\nConfig: {label}\n{'='*60}")

        search_engine = _build_search(
            embedding_model=cfg["embedding"],
            search_mode=cfg["search"],
            chunking=cfg["chunking"],
            rrf_alpha=cfg.get("rrf_alpha"),
            rrf_k=cfg.get("rrf_k"),
        )

        # Use the right chunks_lookup for this chunking strategy
        ck = cfg["chunking"]
        if ck not in _lookup_cache:
            meta = CHUNKING_META.get(ck, CHUNKING_META["section_aware"])
            _lookup_cache[ck] = _load_chunks_lookup(meta["chunks_file"])
        active_lookup = _lookup_cache[ck]

        per_q: list[dict] = []

        for q in tqdm(questions, desc=label, leave=False):
            t0 = time.time()
            results = search_engine.search(
                q["question"],
                mode=cfg["search"],
                chunks_lookup=active_lookup,
            )
            latency = time.time() - t0

            metrics = compute_retrieval_metrics(
                results, q.get("relevant_links", [])
            )
            img_metrics = compute_image_recall(
                results, q.get("relevant_images", [])
            )
            per_q.append({
                "question": q["question"],
                "source_page": q.get("source_page", ""),
                "difficulty": q.get("difficulty", ""),
                "latency": round(latency, 4),
                **metrics,
                **img_metrics,
            })

        all_results[label] = per_q

    # Aggregate
    summary_rows = []
    for label, per_q in all_results.items():
        n = len(per_q)
        agg = {
            "config": label,
            "n": n,
            "recall@5": sum(r["recall@5"] for r in per_q) / n,
            "recall@10": sum(r["recall@10"] for r in per_q) / n,
            "precision@10": sum(r["precision@10"] for r in per_q) / n,
            "mrr": sum(r["mrr"] for r in per_q) / n,
            "image_recall": sum(r["image_recall"] for r in per_q) / n,
            "avg_latency": sum(r["latency"] for r in per_q) / n,
        }
        summary_rows.append(agg)
        logger.info(
            f"  {label}: R@5={agg['recall@5']:.3f}  R@10={agg['recall@10']:.3f}  "
            f"MRR={agg['mrr']:.3f}  ImgR={agg['image_recall']:.3f}"
        )

    return {"axis": axis, "summary": summary_rows, "per_question": all_results}


# ===================================================================
# PHASE 2 - GENERATOR EVALUATION
# ===================================================================

def run_generator(
    questions: list[dict],
    chunks_lookup: dict[str, dict],
    embedding: str,
    search_mode: str,
    chunking: str,
    model_keys: list[str] | None = None,
) -> dict:
    """Run the generator evaluation: best retrieval -> all LLMs.

    Parameters
    ----------
    model_keys : list[str] | None
        Which LLM model keys to evaluate.  Defaults to all in LLM_MODELS.
    """
    from backend.retrieval.llm_client import get_llm_client

    model_keys = model_keys or list(LLM_MODELS.keys())

    # Text is hydrated directly from ChromaDB/SQLite in HybridSearch results;
    # chunks_lookup is not required for the generator phase.

    search_engine = _build_search(
        embedding_model=embedding,
        search_mode=search_mode,
        chunking=chunking,
    )

    # Pre-run retrieval for all questions (same for every LLM)
    logger.info("Running retrieval for all questions ...")
    retrieved: list[tuple[list, str]] = []
    for q in tqdm(questions, desc="Retrieving"):
        results = search_engine.search(
            q["question"], mode=search_mode, chunks_lookup=chunks_lookup
        )
        ctx = _build_context_string(results)
        retrieved.append((results, ctx))

    run_config = {
        "embedding": embedding,
        "search_mode": search_mode,
        "chunking": chunking,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    all_results: dict[str, list[dict]] = {}
    _SAVE_EVERY = 25  # flush to disk every N questions

    for mkey in model_keys:
        info = LLM_MODELS[mkey]
        logger.info(f"\n{'='*60}\nLLM: {info.label} ({info.backend})\n{'='*60}")
        client = get_llm_client(mkey)

        # Rolling checkpoint path — one file per model, overwritten every _SAVE_EVERY questions
        ckpt_path = RESULTS_DIR / f"generator_ckpt_{mkey}.json"

        per_q: list[dict] = []
        for idx, q in enumerate(tqdm(questions, desc=info.label, leave=False)):
            results, ctx = retrieved[idx]
            golden = q.get("answer", "")

            try:
                t0 = time.time()
                resp = client.generate(query=q["question"], context=ctx)
                latency = time.time() - t0

                answer = _strip_thinking(resp.content)
                f1 = compute_token_f1(answer, golden)
                rouge_l = compute_rouge_l(answer, golden)
                cit_f = _citation_faithfulness(answer, results, q.get("source_page", ""))

                per_q.append({
                    "question": q["question"],
                    "source_page": q.get("source_page", ""),
                    "difficulty": q.get("difficulty", ""),
                    "golden_answer": golden,
                    "model_answer": answer,
                    "token_f1": round(f1, 4),
                    "rouge_l": round(rouge_l, 4),
                    "citation_faithfulness": cit_f,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "latency": round(latency, 4),
                })
            except Exception as e:
                logger.error(f"Error with {info.label} on Q{idx}: {e}")
                per_q.append({
                    "question": q["question"],
                    "source_page": q.get("source_page", ""),
                    "difficulty": q.get("difficulty", ""),
                    "golden_answer": golden,
                    "model_answer": f"[ERROR: {e}]",
                    "token_f1": 0.0,
                    "rouge_l": 0.0,
                    "citation_faithfulness": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency": 0.0,
                })

            # Rolling save — overwrite every _SAVE_EVERY questions
            if (idx + 1) % _SAVE_EVERY == 0:
                with open(ckpt_path, "w", encoding="utf-8") as _f:
                    json.dump({"run_config": run_config, "model": info.label,
                               "model_key": mkey, "backend": info.backend,
                               "n_complete": idx + 1, "answers": per_q}, _f,
                              indent=2, ensure_ascii=False)

        # --- BERTScore (batch, post-inference) ---
        try:
            from bert_score import score as _bert_score
            candidates = [r["model_answer"] for r in per_q]
            references = [r["golden_answer"] for r in per_q]
            logger.info(f"Computing BERTScore for {info.label} ({len(candidates)} answers)...")
            _, _, bert_f1 = _bert_score(candidates, references, lang="en", verbose=False)
            for r, bf1 in zip(per_q, bert_f1.tolist()):
                r["bert_score_f1"] = round(bf1, 4)
        except Exception as e:
            logger.warning(f"BERTScore failed for {info.label}: {e} — skipping")
            for r in per_q:
                r["bert_score_f1"] = None

        all_results[mkey] = per_q

        # Final per-model save (with BERTScores, full run config)
        model_out_path = RESULTS_DIR / f"generator_{mkey}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(model_out_path, "w", encoding="utf-8") as f:
            json.dump({"run_config": run_config, "model": info.label,
                       "model_key": mkey, "backend": info.backend,
                       "n_complete": len(per_q), "answers": per_q}, f,
                      indent=2, ensure_ascii=False)
        logger.info(f"Model results saved: {model_out_path}")

    # Aggregate
    summary_rows = []
    for mkey, per_q in all_results.items():
        info = LLM_MODELS[mkey]
        n = len(per_q)
        bert_scores = [r["bert_score_f1"] for r in per_q if r.get("bert_score_f1") is not None]
        cit_scores = [r["citation_faithfulness"] for r in per_q if r.get("citation_faithfulness") is not None]
        agg = {
            "model": info.label,
            "model_key": mkey,
            "backend": info.backend,
            "n": n,
            "avg_f1": sum(r["token_f1"] for r in per_q) / n,
            "avg_rouge_l": sum(r["rouge_l"] for r in per_q) / n,
            "avg_bert_score_f1": sum(bert_scores) / len(bert_scores) if bert_scores else None,
            "citation_faithfulness": sum(cit_scores) / len(cit_scores) if cit_scores else None,
            "citation_faithfulness_n": len(cit_scores),
            "avg_latency": sum(r["latency"] for r in per_q) / n,
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in per_q),
            "total_completion_tokens": sum(r["completion_tokens"] for r in per_q),
        }
        summary_rows.append(agg)
        cit_str = f"{agg['citation_faithfulness']:.3f} ({agg['citation_faithfulness_n']})" if agg["citation_faithfulness"] is not None else "N/A"
        bert_str = f"{agg['avg_bert_score_f1']:.3f}" if agg["avg_bert_score_f1"] is not None else "N/A"
        logger.info(
            f"  {info.label}: F1={agg['avg_f1']:.3f}  ROUGE-L={agg['avg_rouge_l']:.3f}  "
            f"BERTScore={bert_str}  "
            f"CitF={cit_str}  Latency={agg['avg_latency']:.2f}s"
        )

    return {
        "retrieval_config": {
            "embedding": embedding,
            "search_mode": search_mode,
            "chunking": chunking,
        },
        "summary": summary_rows,
        "per_question": all_results,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _write_retriever_report(data: dict, out_dir: Path, ts: str) -> None:
    """Write Markdown summary for a retriever axis run."""
    axis = data["axis"]
    rows = data["summary"]

    lines = [
        f"# Retriever Evaluation - Axis: {axis}",
        f"_Generated: {ts}_\n",
        "| Config | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | Latency |",
        "|--------|----------|-----------|------|-----|------------|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | "
            f"{r['precision@10']:.3f} | {r['mrr']:.3f} | "
            f"{r['image_recall']:.3f} | {r['avg_latency']:.3f}s |"
        )

    md_path = out_dir / f"retriever_{axis}_{ts}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report: {md_path}")


def _write_generator_report(data: dict, out_dir: Path, ts: str) -> None:
    """Write Markdown summary for a generator run."""
    cfg = data["retrieval_config"]
    rows = data["summary"]

    lines = [
        "# Generator Evaluation",
        f"_Retrieval config: {cfg['embedding']} | {cfg['search_mode']} | {cfg['chunking']}_",
        f"_Generated: {ts}_\n",
        "| Model | Backend | Avg F1 | Avg ROUGE-L | Avg BERTScore | Avg Latency | Prompt Tok | Compl Tok |",
        "|-------|---------|--------|-------------|---------------|-------------|------------|-----------||",
    ]
    for r in rows:
        bs = f"{r['avg_bert_score_f1']:.3f}" if r.get("avg_bert_score_f1") is not None else "N/A"
        lines.append(
            f"| {r['model']} | {r['backend']} | {r['avg_f1']:.3f} | "
            f"{r['avg_rouge_l']:.3f} | {bs} | {r['avg_latency']:.2f}s | "
            f"{r['total_prompt_tokens']:,} | {r['total_completion_tokens']:,} |"
        )

    md_path = out_dir / f"generator_{ts}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report: {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ore-acle Evaluation Runner - two-phase ablation framework"
    )
    p.add_argument(
        "--phase",
        choices=["retriever", "generator"],
        required=True,
        help="Evaluation phase",
    )
    p.add_argument(
        "--axis",
        choices=["embedding", "search", "chunking", "rrf"],
        default=None,
        help="Retriever axis to vary (required for --phase retriever)",
    )
    p.add_argument(
        "--embedding",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model for generator phase (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    p.add_argument(
        "--search-mode",
        default=DEFAULT_SEARCH_MODE,
        help=f"Search mode for generator phase (default: {DEFAULT_SEARCH_MODE})",
    )
    p.add_argument(
        "--chunking",
        default=DEFAULT_CHUNKING,
        help=f"Chunking strategy for generator phase (default: {DEFAULT_CHUNKING})",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="LLM model keys to evaluate (default: all). E.g. --models qwen3-0.6b gemini-pro",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions to evaluate (default: all)",
    )
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if args.phase == "retriever" and args.axis is None:
        logger.error("--axis is required for --phase retriever")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    questions = load_dataset(DATASET_PATH, limit=args.limit)
    # Text hydration is now handled on-demand via SQLite inside HybridSearch —
    # no need to load the full chunks.json into RAM upfront.
    chunks_lookup: dict = {}

    if args.phase == "retriever":
        data = run_retriever_axis(args.axis, questions, chunks_lookup)

        # Save JSON
        json_path = RESULTS_DIR / f"retriever_{args.axis}_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON: {json_path}")

        _write_retriever_report(data, RESULTS_DIR, ts)

    elif args.phase == "generator":
        data = run_generator(
            questions=questions,
            chunks_lookup=chunks_lookup,
            embedding=args.embedding,
            search_mode=args.search_mode,
            chunking=args.chunking,
            model_keys=args.models,
        )

        json_path = RESULTS_DIR / f"generator_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON: {json_path}")

        _write_generator_report(data, RESULTS_DIR, ts)


if __name__ == "__main__":
    main()
