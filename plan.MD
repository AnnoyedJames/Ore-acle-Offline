# Ore-acle Offline — Project Plan

> **Goal**: Port the cloud-hosted Ore-acle RAG system to a fully local, modular
> architecture for graduate-level ablation testing, embedding benchmarking, and
> evaluation framework development.

---

## 1. Motivation

The original Ore-acle ("Online") is deployed on Vercel with Pinecone, Supabase,
and Cloudflare R2. That works for a demo, but academic evaluation demands:

- **Reproducibility**: All components must run locally with no external API
  dependencies for the core retrieval pipeline.
- **Ablation testing**: Swap individual components (search strategy, embedding
  model, chunking method, LLM) while holding everything else constant.
- **Metrics**: Automated evaluation with ground-truth datasets and standard
  IR/NLG metrics.
- **Experimentation speed**: No rate limits, no billing surprises, no cold starts.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Next.js Frontend (React)               │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ Chat UI     │  │ Model      │  │ Settings Gear    │  │
│  │ (preserved) │  │ Selector ▼ │  │ (search mode,    │  │
│  │             │  │            │  │  tool calling,   │  │
│  │             │  │            │  │  reasoning)      │  │
│  └─────────────┘  └────────────┘  └──────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP (localhost)
┌───────────────────────▼─────────────────────────────────┐
│                  FastAPI Backend (Python)                 │
│                                                          │
│  /api/chat     — RAG pipeline (configurable)             │
│  /api/models   — List available OpenRouter models        │
│  /api/settings — Read/update runtime config              │
│  /api/image    — Serve local WebP images                 │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Retrieval Pipeline                    │   │
│  │                                                    │   │
│  │  Query → Embed → Search → Rerank → Generate       │   │
│  │           │        │                    │          │   │
│  │           ▼        ▼                    ▼          │   │
│  │      Embedding   ChromaDB           LangChain +   │   │
│  │      Model       + SQLite           OpenRouter    │   │
│  │   (swappable)    FTS5             (model selector)│   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              Evaluation Framework (Python CLI)            │
│                                                          │
│  generate_dataset.py  → Gold QA pairs from wiki chunks   │
│  run_eval.py          → Ablation runner + metrics        │
│                                                          │
│  Metrics: Recall@K, MRR, LLM-as-Judge (answer quality)  │
│  Axes:    search mode × embedding model × chunker × LLM │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Local Storage Stack

| Role | Online (old) | Offline (new) | Why |
|------|-------------|---------------|-----|
| Vectors + metadata | Pinecone (serverless) | **ChromaDB** (local) | Zero setup, Python-native, supports multiple collections for A/B testing |
| Keyword search | Supabase (PostgreSQL tsvector) | **SQLite FTS5** | Single-file DB, blazing fast full-text search, no server process |
| Images | Cloudflare R2 | **Local filesystem** (`data/raw/images/`) | Already have 61k WebP images locally, served via FastAPI static route |
| LLM gateway | DeepSeek API (direct) | **OpenRouter** via LangChain | Single API key → access to dozens of models (DeepSeek, GPT-4o, Claude, Gemini, Llama, etc.) |

---

## 4. Implementation Phases

### Phase 1: Evaluation Framework ← **CURRENT**

Build the eval system *first* so every subsequent change is measurable.

| Step | Description | File(s) |
|------|-------------|---------|
| 1a | Dataset generator: sample chunks, LLM generates grounded QA pairs | `scripts/eval/generate_dataset.py` ✅ |
| 1b | Eval runner: programmatic ablation across search/embed/model axes | `scripts/eval/run_eval.py` |
| 1c | Metrics module: Recall@K, MRR, LLM-as-Judge scoring | `scripts/eval/metrics.py` |
| 1d | Baseline: run current system config, record scores | `data/eval/results/baseline.json` |

### Phase 2: Local Backend (FastAPI + ChromaDB + SQLite)

Replace cloud services with local equivalents.

| Step | Description | File(s) |
|------|-------------|---------|
| 2a | FastAPI server with `/api/chat`, `/api/models`, `/api/image` | `backend/server.py` |
| 2b | ChromaDB integration (create collection, query) | `backend/database/vectorstore.py` |
| 2c | SQLite FTS5 integration (keyword search) | `backend/database/keyword_store.py` |
| 2d | Population script: chunks.json → ChromaDB + SQLite | `scripts/populate_local_db.py` |
| 2e | LangChain + OpenRouter wiring with tool calling | `backend/retrieval/answer.py` |

### Phase 3: Embedding Model Ablation

Benchmark multiple text embedding models.

| Step | Description | File(s) |
|------|-------------|---------|
| 3a | Modular embedding generator (supports E5, BGE-M3, Nomic, Gemini) | `backend/embeddings/generator.py` |
| 3b | Generate embeddings per model → separate ChromaDB collections | `scripts/generate_embeddings.py` |
| 3c | Run eval across embedding models | `scripts/eval/run_eval.py` |

### Phase 4: Image Pipeline Improvements

Deduplicate and optionally embed images.

| Step | Description | File(s) |
|------|-------------|---------|
| 4a | Perceptual hash deduplication (keep highest-res) | `backend/preprocessing/deduplicate_images.py` |
| 4b | Exploratory: CLIP/SigLIP image embeddings | `backend/embeddings/image_embeddings.py` |
| 4c | Multimodal retrieval (text + image vectors) | `backend/retrieval/search.py` |

### Phase 5: Frontend Refactoring

Preserve the UI but point it at the local backend.

| Step | Description | File(s) |
|------|-------------|---------|
| 5a | Proxy API calls to FastAPI `localhost:8000` | `frontend/next.config.js` |
| 5b | Model selector dropdown (VS Code Copilot style) | `frontend/components/ChatInterface.tsx` |
| 5c | Settings gear (search mode, tool calling, reasoning) | `frontend/components/SettingsMenu.tsx` |
| 5d | Remove About page and Vercel-specific code | Various |

### Phase 6: Chunking Ablation

Compare the existing custom chunker vs LangChain splitters.

| Step | Description | File(s) |
|------|-------------|---------|
| 6a | LangChain RecursiveCharacterTextSplitter integration | `backend/preprocessing/chunker.py` |
| 6b | Re-chunk corpus with both methods, populate separate collections | `scripts/generate_chunks.py` |
| 6c | Eval comparison | `scripts/eval/run_eval.py` |

---

## 5. Evaluation Framework Design

### 5.1 Dataset Generation (Automated)

```
┌──────────────────┐     ┌─────────────┐     ┌──────────────────┐
│ chunks.json      │────▶│ Sample N    │────▶│ LLM generates    │
│ (347k chunks)    │     │ random high │     │ 1-2 QA pairs per │
│                  │     │ quality     │     │ chunk (grounded) │
└──────────────────┘     │ chunks      │     └────────┬─────────┘
                         └─────────────┘              │
                                                      ▼
                                            ┌──────────────────┐
                                            │ eval_dataset.json│
                                            │                  │
                                            │ question         │
                                            │ gold_answer      │
                                            │ gold_urls        │
                                            │ source_chunk_id  │
                                            │ source_text      │
                                            └──────────────────┘
```

**Why this works:**
- We already have the source text → the gold answer is grounded by construction.
- The chunk's `page_url` is the primary gold retrieval target.
- The chunk's `related_pages` provide secondary gold URLs.
- Fully automated and infinitely scalable.

### 5.2 Metrics

| Metric | What it measures | How |
|--------|-----------------|-----|
| **Recall@K** | Did the correct source page appear in the top K retrieved chunks? | Compare `gold_urls` against top-K result URLs |
| **MRR** (Mean Reciprocal Rank) | How high did the correct source rank? | 1/rank of first matching gold URL in results |
| **Answer Quality** (LLM-as-Judge) | Is the generated answer faithful and complete? | A judge LLM scores the answer against the gold answer (1-5 scale) |

### 5.3 Ablation Axes

| Axis | Options |
|------|---------|
| Search strategy | Semantic only · Keyword only · Hybrid (RRF) |
| Text embedding model | Multilingual-E5-Large · BGE-M3 · Nomic v1.5 · Gemini |
| Chunking method | Custom recursive (512 tok) · LangChain splitters |
| LLM (generation) | DeepSeek · GPT-4o-mini · Claude Haiku · Gemini Flash · Llama 3.3 |
| Top-K | 3, 5, 10 |

---

## 6. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| OpenRouter as LLM gateway | Single API key → dozens of models. LangChain `ChatOpenAI` wrapper with `base_url` override makes this trivial. |
| ChromaDB over FAISS | ChromaDB supports metadata filtering, multiple collections (one per embedding model), and persistence out of the box. FAISS would need custom serialization. |
| SQLite FTS5 over Whoosh/Elasticsearch | Zero-dependency, single-file, easily embedded in Python. FTS5 is production-grade full-text search. |
| Eval-first development | Building the evaluation framework before changing any component ensures every modification is measurable. Prevents "improvement by vibes." |
| Gemini 2.5 Flash (thinking) for dataset generation | Thinking models self-verify that questions are actually answerable from the text, producing higher quality ground truth. |
| Separate ChromaDB collections per embedding model | Avoids dimension mismatches and allows clean A/B retrieval comparisons without re-indexing. |

---

## 7. Environment Setup

```bash
# Python backend
pip install fastapi uvicorn chromadb langchain langchain-openai openai pydantic python-dotenv

# .env file
OPENROUTER_API_KEY=sk-or-...
```

```bash
# Frontend (unchanged)
cd frontend && npm install && npm run dev
```

---

## 8. File Structure (New/Modified)

```
scripts/
├── eval/
│   ├── generate_dataset.py   # Gold QA pair generator (Gemini 2.5 Flash)
│   ├── run_eval.py           # Ablation runner
│   └── metrics.py            # Recall@K, MRR, LLM-as-Judge
├── populate_local_db.py      # ChromaDB + SQLite population
└── generate_embeddings.py    # Multi-model embedding generation

backend/
├── server.py                 # FastAPI application
├── database/
│   ├── vectorstore.py        # ChromaDB wrapper
│   └── keyword_store.py      # SQLite FTS5 wrapper
├── embeddings/
│   ├── generator.py          # Multi-model embedding generator
│   └── image_embeddings.py   # CLIP/SigLIP experiments
├── preprocessing/
│   ├── chunker.py            # + LangChain splitter option
│   └── deduplicate_images.py # pHash deduplication
└── retrieval/
    ├── search.py             # Configurable hybrid search
    └── answer.py             # LangChain + OpenRouter generation

data/
├── eval/
│   ├── eval_dataset.json     # Generated gold QA pairs
│   └── results/              # Eval run outputs
├── local_db/
│   ├── chroma/               # ChromaDB persistence
│   └── oreacle.sqlite        # SQLite FTS5 database
└── processed/                # (existing) chunks, embeddings, metadata
```
