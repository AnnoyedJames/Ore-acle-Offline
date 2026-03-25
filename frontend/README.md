# Ore-acle Offline Frontend

Next.js 16 App Router application with glassmorphism-themed Minecraft wiki chat interface. It serves as the UI for the offline local RAG system.

## Changes for Offline Mode

The frontend has been adapted to function without direct cloud dependencies:
- **Cloudflare R2**: Removed. Images are served directly from the local `data/raw/images` directory via `/api/image/[hash]`.
- **Pinecone / Supabase**: Removed. The `/api/chat` endpoint is now a simple proxy that forwards requests to the local Python backend (`http://127.0.0.1:8000`).
- **HuggingFace Inference**: Removed. Embeddings are handled by the local backend.

## Features

### Core UI Components
- **Header**: Branding + theme toggle + About link
- **ChatInterface**: Message management, API calls, auto-scroll
- **MessageBubble**: User/assistant styling with citations
- **SourceCard**: Citation tooltips
- **ImageGallery**: Local image thumbnails + expand modal
- **LoadingSpinner**: Minecraft-themed animations

### APIs (Local Proxies)
- **POST /api/chat**: Proxies to `http://127.0.0.1:8000/chat`
- **GET /api/image/[hash]**: Serves local image files directly

## Quick Start

1. Start the Python Backend (FastAPI):
   ```bash
   cd ../backend
   uvicorn api.server:app --reload
   ```

2. Start the Frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. Open http://localhost:3000

## Environment Variables

Only essential UI variables remain in `.env.local` (if any). Cloud API keys have been removed.
