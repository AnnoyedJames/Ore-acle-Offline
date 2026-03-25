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

### Data Assets
- **12,487 HTML pages** (Minecraft Wiki snapshot)
- **61,000+ images** (Local WebP files)
- **Evaluation Dataset**: `data/eval/eval_dataset.json` (1,000 generated QA pairs)

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
- **Generation**: `scripts/eval/generate_dataset.py` creates the Golden Test Set.
- **Benchmarking**: `scripts/eval/run_eval.py` runs the ablation tests.
- **Metrics**: Recall@K, MRR, Answer Relevance.

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
- [ ] Local Vector DB (ChromaDB)
- [ ] Local Keyword DB (SQLite)
- [ ] FastAPI Backend
- [ ] Evaluation Framework

