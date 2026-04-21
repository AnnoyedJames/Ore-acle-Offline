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
import re
import sys
import time
from pathlib import Path
from typing import Optional, Any

import numpy as np

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


# Non-game page types that are false-positive candidates.
# Extend this set if other junk categories appear in the corpus.
_JUNK_PAGE_TYPES = frozenset({
    "novel", "book", "disambiguation", "redirect",
    "talk", "meta", "other",
})

# Pages whose title matches these patterns are also treated as junk.
_JUNK_TITLE_RE = re.compile(
    r"(?i)"
    r"(minecraft:\s*(the\s+(island|voyage|survivors|lost\s+journals|woodsword\s+chronicles|stonesword\s+saga|wither\s+without\s+you))|"
    r"\b(java|bedrock)\s+edition\b|"
    r"^(talk:|user:|minecraft\s+wiki:|template:|java\s+edition\s+\d|bedrock\s+edition\s+\d|"
    r"\d+\.\d+(\.\d+)?(\s+(pre-release|release\s+candidate|snapshot))?))"
)


def compute_fpr(results: list, k: int = 3) -> dict:
    """False Positive Rate at k: fraction of top-k results that are junk pages.

    'Junk' = novel/book/disambiguation/meta page types, or title matches
    the novel title pattern.  A high FPR means the retriever is wasting
    context slots on irrelevant content.
    """
    if not results:
        return {"fpr@3": 0.0, "junk_in_top3": 0}

    top = results[:k]
    junk = 0
    for res in top:
        ptype = getattr(res, "page_type", "") or ""
        title = getattr(res, "page_title", "") or ""
        if ptype.lower() in _JUNK_PAGE_TYPES or _JUNK_TITLE_RE.search(title):
            junk += 1
    return {"fpr@3": junk / len(top), "junk_in_top3": junk}


# Patterns indicating the model could not answer from the retrieved context.
_NO_ANSWER_RE = re.compile(
    r"(?i)(do(es)?\s+not\s+contain|not\s+enough\s+information|"
    r"cannot\s+(find|answer|provide)|"
    r"(the\s+)?source[s]?\s+don'?t\s+(contain|mention|include)|"
    r"i\s+(am\s+)?sorry|no\s+information|"
    r"not\s+mentioned\s+in|cannot\s+be\s+found|"
    r"unable\s+to\s+(find|answer)|"
    r"provided\s+source[s]?\s+do\s+not)"
)


def compute_no_answer_rate(answer: str) -> bool:
    """Return True if the answer text signals a retrieval failure."""
    return bool(_NO_ANSWER_RE.search(answer))


# ---------------------------------------------------------------------------
# Passage-level recall (requires gold_spans in dataset)
# ---------------------------------------------------------------------------
def _token_overlap(a: str, b: str) -> float:
    """Unigram F1 between two strings (case-insensitive)."""
    ta = set(re.sub(r"[^a-z0-9 ]", " ", a.lower()).split())
    tb = set(re.sub(r"[^a-z0-9 ]", " ", b.lower()).split())
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    prec = len(intersection) / len(tb)
    rec = len(intersection) / len(ta)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _span_hit(span_text: str, chunk_text: str, threshold: float = 0.60) -> bool:
    """Return True if *span_text* is found in *chunk_text*.

    First tries exact substring (handles direct quotes).  Falls back to
    token-overlap F1 ≥ threshold for cases where whitespace / unicode
    normalisation differs slightly.
    """
    span_norm = re.sub(r"\s+", " ", span_text.strip().lower())
    chunk_norm = re.sub(r"\s+", " ", chunk_text.strip().lower())
    if span_norm in chunk_norm:
        return True
    return _token_overlap(span_norm, chunk_norm) >= threshold


def compute_passage_recall(
    results: list,
    gold_spans: list[dict],
    multi_hop: bool = False,
    k5: int = 5,
    k10: int = 10,
) -> dict:
    """Passage-level recall using verbatim gold spans.

    For single-page questions (multi_hop=False):
        passage_recall@K = fraction of gold spans covered by at least one
        top-K chunk (OR logic — any span covered counts).

    For multi-hop questions (multi_hop=True):
        passage_recall@K = fraction of distinct hops (page groups) where
        at least one span from that hop was covered (AND logic — all hops
        needed for full score; partial hop coverage reported as hop_coverage).

    Falls back gracefully to None when gold_spans is empty.
    """
    if not gold_spans:
        return {
            "passage_recall@5": None,
            "passage_recall@10": None,
            "hop_coverage": None,
        }

    chunks_at_5 = [getattr(r, "text", "") or "" for r in results[:k5]]
    chunks_at_10 = [getattr(r, "text", "") or "" for r in results[:k10]]

    if not multi_hop:
        # OR logic: fraction of spans hit by any top-K chunk
        hits5 = sum(
            1 for s in gold_spans
            if any(_span_hit(s["text"], c) for c in chunks_at_5)
        )
        hits10 = sum(
            1 for s in gold_spans
            if any(_span_hit(s["text"], c) for c in chunks_at_10)
        )
        n = len(gold_spans)
        return {
            "passage_recall@5": hits5 / n,
            "passage_recall@10": hits10 / n,
            "hop_coverage": None,  # N/A for single-hop
        }
    else:
        # AND logic: group spans by hop, require all hops to be covered
        hops: dict[int, list[str]] = {}
        for s in gold_spans:
            hop = s.get("hop", 1)
            hops.setdefault(hop, []).append(s["text"])

        n_hops = len(hops)
        hops_hit_at_5 = sum(
            1 for hop_spans in hops.values()
            if any(_span_hit(sp, c) for sp in hop_spans for c in chunks_at_5)
        )
        hops_hit_at_10 = sum(
            1 for hop_spans in hops.values()
            if any(_span_hit(sp, c) for sp in hop_spans for c in chunks_at_10)
        )
        return {
            "passage_recall@5": hops_hit_at_5 / n_hops,
            "passage_recall@10": hops_hit_at_10 / n_hops,
            "hop_coverage": hops_hit_at_10 / n_hops,
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

# Minimal fields needed for retrieval metrics (page_url, images).
# Omitting "text" avoids loading ~1.5 GB of text into memory.
_LOOKUP_FIELDS = frozenset(
    {"chunk_id", "page_title", "page_url", "section_heading",
     "images", "chunk_type", "page_type", "infobox"}
)


def _load_chunks_lookup(path: Path | None = None) -> dict[str, dict]:
    """Stream a chunks JSON file into a lightweight {chunk_id: chunk_dict}.

    Uses ``ijson`` for streaming to avoid loading the full multi-GB file into
    memory.  Only the fields required for retrieval metrics are kept.
    Falls back to standard ``json.load`` if ``ijson`` is not installed.
    """
    cpath = path or settings.chunks_file
    if not cpath.exists():
        logger.warning(f"{cpath} not found")
        return {}

    logger.info(f"Loading chunks lookup from {cpath} (streaming) ...")
    try:
        import ijson  # type: ignore
        lookup: dict[str, dict] = {}
        with open(cpath, "rb") as f:
            for chunk in ijson.items(f, "item"):
                cid = chunk.get("chunk_id", "")
                if cid:
                    lookup[cid] = {k: v for k, v in chunk.items() if k in _LOOKUP_FIELDS}
        logger.info(f"Loaded {len(lookup)} chunk entries (streaming)")
        return lookup
    except ImportError:
        logger.warning("ijson not installed — falling back to json.load (may OOM on large files)")
    except Exception as e:
        logger.warning(f"ijson streaming failed ({e}) — falling back to json.load")

    with open(cpath, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: {k: v for k, v in c.items() if k in _LOOKUP_FIELDS}
            for c in chunks}


# ===================================================================
# PHASE 1 - RETRIEVER EVALUATION
# ===================================================================

_EVAL_CACHE_DIR = Path("data/eval")


def _load_or_build_query_cache(
    questions: list[dict],
    embedder: Any,
    model_id: str,
) -> dict[str, Any]:
    """Return {question_text: np.ndarray | None} for every question.

    On the first call for a given model the embeddings are computed and saved
    to ``data/eval/query_embeddings_<safe_model>.npy`` +
    ``data/eval/query_embeddings_<safe_model>_ids.json``.
    Subsequent calls (or subsequent eval runs) load from disk — no API calls.
    """
    safe = model_id.replace("/", "_").replace("-", "_")
    npy_path = _EVAL_CACHE_DIR / f"query_embeddings_{safe}.npy"
    ids_path = _EVAL_CACHE_DIR / f"query_embeddings_{safe}_ids.json"

    texts = [q["question"] for q in questions]

    # --- load from disk if available ---
    if npy_path.exists() and ids_path.exists():
        cached_ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))
        vecs = np.load(npy_path, allow_pickle=False)
        if len(cached_ids) == len(vecs):
            result: dict[str, Any] = dict(zip(cached_ids, vecs))
            # Fill missing questions (e.g. dataset grew since last cache run)
            missing = [t for t in texts if t not in result]
            if not missing:
                logger.info(
                    f"Query embedding cache hit ({len(cached_ids)} vectors) for model '{model_id}'"
                )
                return result
            logger.info(
                f"Partial cache hit for '{model_id}': {len(cached_ids)} cached, "
                f"{len(missing)} new questions need embedding"
            )
            texts_to_embed = missing
        else:
            logger.warning("Cache size mismatch — rebuilding from scratch")
            result = {}
            texts_to_embed = texts
    else:
        result = {}
        texts_to_embed = texts

    # --- embed missing questions ---
    logger.info(f"Embedding {len(texts_to_embed)} questions for model '{model_id}' ...")
    skip_count = 0
    for txt in tqdm(texts_to_embed, desc=f"  embed [{safe}]", leave=False):
        try:
            result[txt] = embedder.embed_query(txt)
        except Exception as e:
            logger.warning(f"Embedding failed for '{txt[:60]}': {e} — skipped")
            result[txt] = None
            skip_count += 1

    if skip_count:
        logger.warning(
            f"{skip_count}/{len(texts_to_embed)} questions failed embedding for '{model_id}'"
        )

    # --- persist to disk (only the successfully embedded questions) ---
    all_texts = list(result.keys())
    valid_mask = [v is not None for v in result.values()]
    valid_texts = [t for t, ok in zip(all_texts, valid_mask) if ok]
    valid_vecs = [result[t] for t in valid_texts]

    if valid_vecs:
        _EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, np.array(valid_vecs, dtype=np.float32))
        ids_path.write_text(json.dumps(valid_texts), encoding="utf-8")
        logger.info(
            f"Saved {len(valid_vecs)} query embeddings to {npy_path} "
            f"(skipped {skip_count} failed)"
        )

    return result


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

    # Per-model disk-backed query embedding cache.
    # Key: model_id  Value: {question_text: np.ndarray | None}
    # Populated once per model, persisted to data/eval/ so subsequent
    # eval runs skip the API call entirely.
    _embed_cache: dict[str, dict[str, Any]] = {}

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

        # Resolve per-model query embedding cache (disk-backed).
        # Keyword-only configs need no embeddings.
        query_vec_cache: dict[str, Any] = {}  # question_text -> np.ndarray | None
        if cfg["search"] != "keyword":
            model_id = cfg["embedding"]
            if model_id not in _embed_cache:
                _embed_cache[model_id] = _load_or_build_query_cache(
                    questions, search_engine.embedder, model_id
                )
            query_vec_cache = _embed_cache[model_id]

        per_q: list[dict] = []

        for q in tqdm(questions, desc=label, leave=False):
            qtext = q["question"]
            qvec = query_vec_cache.get(qtext) if cfg["search"] != "keyword" else None

            # Skip questions where embedding failed — don't pollute metrics with 0s
            if cfg["search"] != "keyword" and qvec is None:
                per_q.append({
                    "question": qtext,
                    "source_page": q.get("source_page", ""),
                    "difficulty": q.get("difficulty", ""),
                    "multi_hop": q.get("multi_hop", False),
                    "latency": 0.0,
                    "skipped": True,
                    "recall@5": None, "recall@10": None,
                    "precision@10": None, "mrr": None, "image_recall": None,
                    "fpr@3": None, "junk_in_top3": None,
                    "passage_recall@5": None, "passage_recall@10": None,
                    "hop_coverage": None,
                })
                continue

            t0 = time.time()
            results = search_engine.search(
                qtext,
                mode=cfg["search"],
                chunks_lookup=active_lookup,
                query_vec=qvec,
            )
            latency = time.time() - t0

            metrics = compute_retrieval_metrics(
                results, q.get("relevant_links", [])
            )
            img_metrics = compute_image_recall(
                results, q.get("relevant_images", [])
            )
            fpr_metrics = compute_fpr(results, k=3)
            passage_metrics = compute_passage_recall(
                results,
                q.get("gold_spans", []),
                multi_hop=q.get("multi_hop", False),
            )
            per_q.append({
                "question": qtext,
                "source_page": q.get("source_page", ""),
                "difficulty": q.get("difficulty", ""),
                "multi_hop": q.get("multi_hop", False),
                "latency": round(latency, 4),
                "skipped": False,
                **metrics,
                **img_metrics,
                **fpr_metrics,
                **passage_metrics,
            })

        all_results[label] = per_q

    # Aggregate
    summary_rows = []
    for label, per_q in all_results.items():
        # Exclude skipped questions (embedding failures) from metric averages
        scored = [r for r in per_q if not r.get("skipped", False)]
        n_total = len(per_q)
        n = len(scored)
        n_skipped = n_total - n
        if n == 0:
            logger.warning(f"Config '{label}': all {n_total} questions skipped — no metrics available")
            continue
        if n_skipped:
            logger.warning(f"Config '{label}': {n_skipped}/{n_total} questions skipped (embedding failure), metrics over {n} questions")
        # Separate passage recall — only average over entries that have gold_spans
        pr_scored = [r for r in scored if r.get("passage_recall@10") is not None]
        agg = {
            "config": label,
            "n": n,
            "n_skipped": n_skipped,
            "recall@5": sum(r["recall@5"] for r in scored) / n,
            "recall@10": sum(r["recall@10"] for r in scored) / n,
            "precision@10": sum(r["precision@10"] for r in scored) / n,
            "mrr": sum(r["mrr"] for r in scored) / n,
            "image_recall": sum(r["image_recall"] for r in scored) / n,
            "fpr@3": sum(r["fpr@3"] for r in scored) / n,
            "passage_recall@5": sum(r["passage_recall@5"] for r in pr_scored) / len(pr_scored) if pr_scored else None,
            "passage_recall@10": sum(r["passage_recall@10"] for r in pr_scored) / len(pr_scored) if pr_scored else None,
            "avg_latency": sum(r["latency"] for r in scored) / n,
        }
        summary_rows.append(agg)
        pr10_str = f"{agg['passage_recall@10']:.3f}" if agg["passage_recall@10"] is not None else "N/A"
        logger.info(
            f"  {label}: R@5={agg['recall@5']:.3f}  R@10={agg['recall@10']:.3f}  "
            f"MRR={agg['mrr']:.3f}  ImgR={agg['image_recall']:.3f}  "
            f"FPR@3={agg['fpr@3']:.3f}  PassR@10={pr10_str}"
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
                no_answer = compute_no_answer_rate(answer)

                per_q.append({
                    "question": q["question"],
                    "source_page": q.get("source_page", ""),
                    "difficulty": q.get("difficulty", ""),
                    "golden_answer": golden,
                    "model_answer": answer,
                    "token_f1": round(f1, 4),
                    "rouge_l": round(rouge_l, 4),
                    "citation_faithfulness": cit_f,
                    "no_answer": no_answer,
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
            "no_answer_rate": sum(1 for r in per_q if r.get("no_answer")) / n,
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
            f"CitF={cit_str}  NAR={agg['no_answer_rate']:.3f}  Latency={agg['avg_latency']:.2f}s"
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
        "| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |",
        "|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|",
    ]
    for r in rows:
        skipped = r.get("n_skipped", 0)
        skip_str = f"{skipped}" if skipped else "0"
        fpr = r.get("fpr@3", 0.0) or 0.0
        pr10 = r.get("passage_recall@10")
        pr10_str = f"{pr10:.3f}" if pr10 is not None else "N/A"
        lines.append(
            f"| {r['config']} | {r['n']} | {skip_str} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | "
            f"{r['precision@10']:.3f} | {r['mrr']:.3f} | "
            f"{r['image_recall']:.3f} | {pr10_str} | {fpr:.3f} | {r['avg_latency']:.3f}s |"
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
        "| Model | Backend | Avg F1 | Avg ROUGE-L | Avg BERTScore | CitF | NAR | Avg Latency | Prompt Tok | Compl Tok |",
        "|-------|---------|--------|-------------|---------------|------|-----|-------------|------------|-----------||",
    ]
    for r in rows:
        bs = f"{r['avg_bert_score_f1']:.3f}" if r.get("avg_bert_score_f1") is not None else "N/A"
        cf = r.get("citation_faithfulness")
        cit_str = f"{cf:.3f}" if cf is not None else "N/A"
        nar = r.get("no_answer_rate", 0.0) or 0.0
        lines.append(
            f"| {r['model']} | {r['backend']} | {r['avg_f1']:.3f} | "
            f"{r['avg_rouge_l']:.3f} | {bs} | {cit_str} | {nar:.3f} | {r['avg_latency']:.2f}s | "
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
