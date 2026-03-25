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
| Text cleaner (HTML → JSON) | ✅ Complete |
| Intelligent chunking | ✅ Complete (~347k chunks) |
| **Vector Database** | 🚧 Migrating to ChromaDB (Local) |
| **Keyword Search** | 🚧 Migrating to SQLite FTS5 (Local) |
| **Image Hosting** | ✅ Local Filesystem (`data/raw/images`) |
| **Backend API** | 🚧 Migrating to FastAPI (`local_server.py`) |
| **RAG Pipeline** | 🚧 Refactoring for Local Execution |
| **Frontend UI** | ✅ Adapted for Local API |

---

## 📁 Project Structure

```
Ore-acle-Offline/
├── backend/
│   ├── api/
│   │   └── server.py              # FastAPI server (Offline Backend)
│   ├── scraper/
│   │   ├── wiki_scraper.py        # Fetches HTML pages
│   │   └── image_downloader.py    # Downloads images
│   ├── preprocessing/
│   │   ├── text_cleaner.py        # HTML → structured JSON
│   │   ├── image_processor.py     # PNG → WebP conversion
│   │   └── chunker.py             # Optimized text splitting
│   ├── embeddings/
│   │   └── generator.py           # Embedding generation (Multi-model support)
│   ├── retrieval/
│   │   ├── search.py              # Hybrid search (ChromaDB + SQLite FTS5)
│   │   └── answer.py              # Retrieval & Answer generation
│   └── config/
│       └── settings.py            # Local configuration
├── data/
│   ├── raw/                       # Scraped HTML & images
│   ├── processed/                 # Metadata, chunks, embeddings
│   ├── chroma_db/                 # Local Vector DB storage
│   └── sqlite_fts.db              # Local Keyword Search DB
├── frontend/                      # Next.js 16 App Router UI
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
