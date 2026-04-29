# Ore-acle Offline — Copilot Instructions

## Project Overview
"Ore-acle Offline" is a local port of the original Minecraft Wiki RAG system. The goal is to replicate the full hybrid search pipeline (Semantic + Keyword) using **strictly local technologies** to enable ablation studies and benchmarking.

## Key Architecture Changes (Cloud → Local)
- **Vector DB**: Pinecone → **ChromaDB** (Persistent local storage)
- **Keyword DB**: Supabase → **SQLite FTS5** (Local full-text search)
- **Image Hosting**: Cloudflare R2 → **Local Filesystem** (`data/raw/images`)
- **Backend**: Vite Frontend App → **FastAPI Python Server** (`backend/api/server.py`)
- **LLM**: DeepSeek API → **OpenRouter / Local LLM** (via LangChain)

## Architecture & Data Flow
```
Wiki Scraper → data/raw/html/
      ↓
Text Cleaner → data/processed/metadata.json
      ↓
   Chunker   → data/processed/chunks.json (and chunks_langchain.json)
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
6. **MCUI Crafting Grids / Recipes Format**: Minecraft Wiki visual CSS grids (like crafting tables or furnaces) are parsed by the `text_cleaner.py` into spatial arrays in the metadata text representation. Example: `[Crafting Recipe: [., ., .] [Iron Ingot, ., Iron Ingot] [., Iron Ingot, .] -> Bucket]`. When generating UI, prompting the LLM, or evaluating, be aware that this text format is the exact representation present in chunks and DBs for recipes.


### Data Assets
- **12,487 HTML pages** (Minecraft Wiki snapshot)
- **61,000+ images** (Local WebP files)
- **Evaluation Dataset**: `data/eval/questionset.json` (333 pairs — 120 source pages, 3 Q/A per page at easy/medium/hard, plus manually seeded adversarial queries)
- **ChromaDB**: `data/chroma_db/` — primary collection `chunks_nomic_ai_nomic_embed_text_v1_5` (94,382 chunks, nomic-embed-text-v1.5, section_aware)
- **SQLite FTS5**: `data/sqlite_fts.db` — 94,404 rows (section_aware)

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
  - Phase 1 RETRIEVER: `--axis search` (semantic/keyword/hybrid), `--axis rrf` (alpha sweep), `--axis embedding`, `--axis chunking`
  - Phase 2 GENERATOR: `--phase generator` — tests 4 LLMs (Gemma 4 e2B/e4B via Ollama, Gemma 4 31B + Gemini Flash Lite via OpenRouter)
- **Metrics**: Recall@5, Recall@10, Precision@10, MRR, Image Recall, Token F1, ROUGE-L, BERTScore F1.
- **Results dir**: `data/eval/results/`


**Narrative (333-question final eval)**: All retriever and generator axes are complete.
- **Search axis**: Semantic (MRR=0.625, R@10=0.514) outperforms Hybrid (MRR=0.614, R@10=0.512) on MRR; Hybrid marginally leads R@10 in the RRF sweep (0.516 at α=0.80). Both far exceed Keyword (OOTB MRR=0.425, Custom MRR=0.513). Hybrid is the operational mode for its image recall balance.
- **RRF sweep**: α=0.80 confirmed optimal (MRR=0.615); semantic endpoint (α=1.0) achieves highest MRR=0.625 but lower image recall.
- **Embedding axis**: nomic-embed-text-v1.5 wins all metrics (MRR=0.612, R@10=0.519, Img Rcl=0.265, 0.08s latency). bge-m3 is close (MRR=0.605). NOTE: The embedding axis eval runs bge-m3 twice — once locally (SentenceTransformers) and once via OpenRouter API. The two rows appear with different case ("BAAI/bge-m3" vs "baai/bge-m3"). Results should be identical; latency is NOT compared on this axis.
- **Chunking axis**: section_aware definitively wins (MRR=0.612, R@10=0.516). The langchain result (≈0) in the 333-question eval is a collection mismatch artifact, not a quality measurement; prior 305-question eval confirmed section_aware MRR=0.658 vs langchain MRR=0.614.
- **Generator axis**: Gemma 4 31B (F1=0.290, BERTScore=0.844, CitF=0.973) and Gemini Flash Lite (F1=0.288, BERTScore=0.844, CitF=0.983) tie on quality; Gemini is 6.5× faster (3.1s vs 20.7s). Gemma 4 e2B is capable for local use (F1=0.244, BERTScore=0.711). Gemma 4 e4B anomalously underperforms (BERTScore=0.310).
**Completed eval axes**: Search ✅, RRF sweep ✅, Embedding ✅, Chunking ✅, Generator ✅

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
- [x] Local Vector DB (ChromaDB) — 94,382 chunks, `chunks_nomic_ai_nomic_embed_text_v1_5` (section_aware, nomic)
- [x] Local Keyword DB (SQLite FTS5) — 94,404 rows, OR-query semantics, BM25 normalization
- [x] Evaluation Framework — `run_eval.py` two-phase ablation
- [x] Gold Questionset — 333 pairs at `data/eval/questionset.json` (120 source pages)
- [x] Search axis eval (333q) — Semantic MRR=0.625, Hybrid MRR=0.614, Keyword Custom MRR=0.513, Keyword OOTB MRR=0.425
- [x] RRF alpha sweep (333q) — α=0.80 optimal (MRR=0.615, R@10=0.516); α=0.90 highest MRR=0.617
- [x] Chunking axis eval (333q) — section_aware MRR=0.612, R@10=0.516 (langchain ≈0 due to collection mismatch)
- [x] Embedding axis eval (333q) — nomic-embed-text-v1.5 MRR=0.612 wins; bge-m3 MRR=0.605; e5-large MRR=0.508
- [x] Generator eval (333q) — Gemma 4 31B F1=0.290/BERTScore=0.844/CitF=0.973; Gemini Flash Lite F1=0.288/BERTScore=0.844/CitF=0.983; e2B F1=0.244; e4B F1=0.108
- [x] Weighted RRF — `rrf_alpha=0.80`, `rrf_k=20` (locked)
- [x] FastAPI Backend
- [x] Paper — `latex_paper/main.tex` (updated with 333-question results)

