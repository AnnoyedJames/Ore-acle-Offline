# ⛏️ Ore-acle Offline

**Your Minecraft Expert** — a fully local, offline-capable Retrieval-Augmented Generation (RAG) system that answers questions about Minecraft using knowledge sourced from the [Minecraft Wiki](https://minecraft.wiki).

---

## What Is Ore-acle Offline?

Ore-acle Offline is a port of the original cloud-native RAG system to a modular, local architecture. It is designed for ablation testing, benchmarking, and privacy-first usage without reliance on external cloud services like Pinecone, Supabase, or Cloudflare.

### Key Goals

- **Fully Offline**: Runs entirely on your local machine using local databases (ChromaDB, SQLite).
- **Modular Architecture**: Swap embedding models, chunking strategies, and search algorithms easily for benchmarking.
- **Accurate Answers**: Grounded in real wiki content, identical to the online version.
- **Source Transparency**: Every answer links back to its wiki sources with verbatim quotes.

> **Note:** Due to storage limitations, the data (`data/raw`, `data/processed`) are gitignored. The full datasets can be regenerated locally using the provided ingestion scripts.

---

## 🏗️ Project Status

> 🚧 **Active development** — Transitioning from Cloud to Local Stack.

| Component | Status |
|---|---|
| Wiki HTML scraper | ✅ Complete (12,487 pages) |
| Wiki image downloader | ✅ Complete (61,248 images) |
| Image processing (WebP conversion) | ✅ Complete |
| Base section-aware Chunking | ✅ Complete |
| LangChain chunking (Ablation target) | ✅ Complete (`data/processed/chunks_langchain.json`, 78,172 chunks) |
| Text cleaner (HTML → JSON) | ✅ Complete (Handles complex visual Crafting Recipes/Spriting via MCUI extraction + prompt enforced markdown tables) |
| Intelligent chunking (section-aware, improved) | ✅ Complete — 94,404 chunks (navigation sections removed, merge threshold raised) |
| **Vector Database** | ✅ ChromaDB — `chunks_nomic_ai_nomic_embed_text_v1_5` (94,382) + `chunks_nomic_ai_nomic_embed_text_v1_5__langchain` (78,161) |
| **Keyword Search** | ✅ SQLite FTS5 — 94,404 rows (section_aware, nomic), OR semantics |
| **Image Hosting** | ✅ Local Filesystem (`data/raw/images`, 61k WebP) |
| **Backend API** | 🚧 FastAPI (`backend/api/server.py`) |
| **Evaluation Framework** | ✅ Two-phase ablation (`run_eval.py`) |
| **Gold Questionset** | ✅ 305 Q/A pairs (`data/eval/questionset.json`) |
| **Search Axis Eval** | ✅ Done — Keyword search (R@10=0.472) dramatically outperforms both Semantic (0.081) and Hybrid (0.085) |
| **RRF Alpha Sweep** | ✅ Done — α=0.80 optimal (k=20); default locked in `settings.py` |
| **Embedding Axis Eval** | 🔄 nomic ✅ ingested; e5-large + gemini ingests pending |
| **Chunking Axis Eval** | ✅ Done — Custom `section_aware` chunker (R@10=0.085 in hybrid) definitively defeats standard `langchain` chunker (R@10=0.050) |
| **Generator Eval** | 🔄 4 LLMs (Gemma4 e2B/e4B/31B + Gemini Flash Lite), keyword mode |
| **Frontend UI** | ✅ Migrated to Vite + React (Offline, API-ready) |

---

## 📁 Project Structure

```
Ore-acle-Offline/
├── archive/                       # Archived logs, scripts, and evaluations
├── backend/
│   ├── api/
│   │   └── server.py              # FastAPI server (Offline Backend)
│   ├── scraper/
│   │   ├── wiki_scraper.py        # Fetches HTML pages
│   │   └── image_downloader.py    # Downloads images
│   ├── preprocessing/
│   │   ├── text_cleaner.py        # MCUI Grid Parsing & HTML → JSON
│   │   ├── image_processor.py     # PNG → WebP conversion
│   │   └── chunker.py             # Optimized text splitting
│   ├── embeddings/
│   │   └── generator.py           # Embedding generation (Multi-model support)
│   ├── retrieval/
│   │   ├── search.py              # Hybrid search + Auto-Crafting logic
│   │   └── answer.py              # Retrieval & Answer generation (Markdown tables)
│   └── config/
│       └── settings.py            # Local configuration (Offline HF enabled)
├── data/
│   ├── raw/                       # Scraped HTML & images
│   ├── processed/                 # Metadata, chunks, embeddings
│   ├── chroma_db/                 # Local Vector DB storage
│   └── sqlite_fts.db              # Local Keyword Search DB
├── frontend/                      # Vite + React UI with Tailwind Typography
├── scripts/
│   ├── eval/                      # Evaluation & Benchmarking
│   │   ├── generate_dataset.py    # Gold-standard QA generation
│   │   └── run_eval.py            # Metrics calculation
│   └── ingest/                    # Data ingestion scripts
└── requirements.txt
```

---

## 🛠️ Tech Stack

### Data Pipeline
- **Language:** Python 3.11+
- **Scraping:** Requests, BeautifulSoup4
- **Image Processing:** Pillow (Perceptual Hashing for deduplication)

### Offline Backend (New)
- **Framework:** FastAPI
- **Vector Database:** ChromaDB (Local, persistent)
- **Keyword Search:** SQLite (FTS5 module)
- **Image Serving:** Direct local file serving
- **Embeddings:** `sentence-transformers` (Local execution)
- **LLM Gateway:** OpenRouter (for swapping models) or Local LLM (Ollama/LM Studio)

### Frontend
- **Framework:** Next.js 16
- **Styling:** Tailwind CSS (Minecraft theme)
- **API:** Proxies to local Python backend
