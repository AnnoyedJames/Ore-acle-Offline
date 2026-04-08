# Ore-acle Offline — Copilot Instructions

## Project Overview
"Ore-acle Offline" is a local port of the original Minecraft Wiki RAG system. The goal is to replicate the full hybrid search pipeline (Semantic + Keyword) using **strictly local technologies** to enable ablation studies and benchmarking.

## Key Architecture Changes (Cloud → Local)
- **Vector DB**: Pinecone → **ChromaDB** (Persistent local storage)
- **Keyword DB**: Supabase → **SQLite FTS5** (Local full-text search)
- **Image Hosting**: Cloudflare R2 → **Local Filesystem** (`data/raw/images`)
- **Backend**: Next.js Serverless → **FastAPI Python Server** (`backend/api/server.py`)
- **LLM**: DeepSeek API → **OpenRouter / Local LLM** (via LangChain)

## Architecture & Data Flow
```
Wiki Scraper → data/raw/html/
      ↓
Text Cleaner → data/processed/metadata.json
      ↓
   Chunker   → data/processed/chunks.json
      ↓
  Embedder   → data/processed/embeddings/*.npy (Nomic/E5/BGE)
      ↓
  Ingestor   → ChromaDB (Vectors) + SQLite (Keyword Index)
      
[Runtime Query Flow]
Frontend (/api/chat) → Proxy → Local FastAPI Server
      ↓
Search Module (Hybrid)
      ├→ ChromaDB query (Semantic)
      └→ SQLite FTS5 query (Keyword)
      ↓
Reciprocal Rank Fusion (RRF)
      ↓
LLM Generation (OpenRouter/Local) → Response
```

### Hybrid Storage Strategy (Offline)
1. **ChromaDB**: Stores embeddings and essential metadata (chunk_id, page_title). Used for semantic retrieval.
2. **SQLite (FTS5)**: Stores the full text index for keyword search (`bm25` equivalent).
3. **Local Filesystem**: Stores the actual text chunks (`chunks.json`) and images.


## Data Schema Quirks & Joins (CRITICAL)

When working with data processing scripts or evaluations, be aware of the following dataset idiosyncrasies to prevent 'hallucinating' missing data or failing array intersections:

1. **metadata.json**: Tracks the original parsed HTML pages. The images array native to a page here includes *every single URL requested* (including 16x16 UI icons, tiny sprites, etc.).
2. **image_metadata.json**: Tracks the images that were *actually downloaded*. The scraper actively strips out small/UI images. Its files are saved locally as data/raw/images/[image_hash].webp.
3. **Array Match Limitations**: Do NOT rely blindly on the source_pages array in image_metadata.json to find all images for a current page. That array is truncated (cap of 18) by the scraping script. 
4. **URL Normalization (The Join Key)**: To correctly intersect a page's requested images with the actual downloaded dataset, you MUST match them via URL. However, the URLs contain encoding inconsistencies and tracking params. You must normalize both sides using Python:
   urllib.parse.unquote(url.split('?')[0].split('#')[0])

5. **SQLite FTS5 Query Semantics**: The keyword search uses OR logic between terms (terms ≤ 2 chars filtered as stopwords). Default FTS5 AND semantics returns 0 results for natural language queries. This fix lives in `SQLiteStore.search()` in `backend/database/local_stores.py`.


### Data Assets
- **12,487 HTML pages** (Minecraft Wiki snapshot)
- **61,000+ images** (Local WebP files)
- **Evaluation Dataset**: `data/eval/questionset.json` (300 pairs — 100 pages × 3 Q/A at easy/medium/hard)
- **ChromaDB**: `data/chroma_db/` — collection `chunks_baai_bge_m3`, 121,080 chunks (BAAI/bge-m3 via OpenRouter API)
- **SQLite FTS5**: `data/sqlite_fts.db` — 121,618 rows

### RAG Citation Design
Maintains the **NotebookLM-style citations**:
- Verbatim source text quoting.
- Explicit page title and URL linking.
- Rich content extraction (Infoboxes + Images).

## Module Conventions

### Configuration
All settings are local-first.
- **Config File**: `backend/config/settings.py`
- **Secrets**: `.env` (Only `OPENROUTER_API_KEY` required, others optional).

### Testing & Evaluation
- **Generation**: `scripts/eval/generate_questionset.py` creates the Golden Test Set (Gemini Flash Lite via OpenRouter, two-pass image selection with ijson streaming).
- **Benchmarking**: `scripts/eval/run_eval.py` — two-phase ablation framework.
  - Phase 1 RETRIEVER: `--axis search` (semantic/keyword/hybrid), `--axis embedding`, `--axis chunking`
  - Phase 2 GENERATOR: `--phase generator` — tests all 5 LLMs
- **Metrics**: Recall@5, Recall@10, Precision@10, MRR, Image Recall, Token F1, ROUGE-L.
- **Results dir**: `data/eval/results/`

### Ablation Results (April 2026)
**Search axis** (300-pair questionset, baai/bge-m3, section_aware chunking):
| Mode | MRR | R@5 | Img Recall | Latency |
|------|-----|-----|------------|---------|
| Semantic | **0.620** | 0.444 | **0.126** | 0.807s |
| Keyword (OR) | 0.428 | 0.360 | 0.061 | **0.067s** |
| Hybrid | 0.575 | 0.445 | 0.106 | 0.865s |

**Finding**: Hybrid underperforms pure semantic. The keyword OR-expansion introduces noisy RRF matches that dilute high-confidence semantic results. **Winning config: `semantic` mode.**

**Remaining eval axes**: Embedding axis (needs 3 new ingests), Chunking axis (needs langchain ingest), Generator eval (5 LLMs, uses semantic mode).

## Code Style
- **Python**: Typed (`mypy` compliant), `black` formatted.
- **Frontend**: TypeScript, Tailwind CSS.
- **Imports**: Absolute imports from `backend` root.

## Mobile Responsive Design
Same constraints as original:
- 12px base font on mobile.
- `overflow-wrap: anywhere` for wide text.
- Glassmorphism UI.

## Implementation Status
- [x] Data Ingestion (Scraper/Cleaner)
- [x] Frontend UI (Local Proxy)
- [x] Local Vector DB (ChromaDB) — 121,080 chunks, `chunks_baai_bge_m3`
- [x] Local Keyword DB (SQLite FTS5) — 121,618 rows, OR-query semantics
- [x] Evaluation Framework — `run_eval.py` two-phase ablation
- [x] Gold Questionset — 300 pairs at `data/eval/questionset.json`
- [x] Search axis eval — Semantic wins (MRR=0.620)
- [ ] FastAPI Backend
- [ ] Embedding axis eval (needs nomic/e5-large/gemini-embedding ingests)
- [ ] Chunking axis eval (needs langchain ingest)
- [ ] Generator eval (5 LLMs, `--phase generator --search-mode semantic`)

