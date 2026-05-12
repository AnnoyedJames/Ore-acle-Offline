import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

from backend.retrieval.search import HybridSearch
from backend.retrieval.answer import AnswerGenerator, GeneratorConfig
from backend.config.settings import LLM_MODELS

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Ore-acle Offline Backend")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allow CORS for Vite dev server and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve local images via static file route
# answer.py generates URLs as /api/image/{hash} — Vite proxy strips /api → /image/{hash}
app.mount("/image", StaticFiles(directory="data/raw/images"), name="images")

# Global instances (lazy loaded or initialized here)
search_engine = HybridSearch()
# We instantiate AnswerGenerator per request to handle different configs


@app.on_event("startup")
async def warmup():
    """Warm up ChromaDB and the Nomic embedding model so the first query is fast."""
    import numpy as np

    # 1. ChromaDB collection warm-up
    logger.info("Warming up ChromaDB collection...")
    try:
        dummy_vec = np.zeros(1024, dtype=np.float32)  # BAAI/bge-m3 dim
        search_engine.chroma.query(dummy_vec, n_results=1)
        logger.info("ChromaDB warm-up complete.")
    except Exception as e:
        logger.warning(f"ChromaDB warm-up failed (non-fatal): {e}")

    # 2. Nomic embedding model warm-up (loads weights into GPU/CPU memory)
    logger.info("Preloading Nomic embedding model...")
    try:
        _ = search_engine.embedder  # triggers lazy model load
        logger.info("Nomic embedding model loaded.")
    except Exception as e:
        logger.warning(f"Nomic model preload failed (non-fatal): {e}")

class Message(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[Message]
    images: Optional[List[str]] = None
    model: str = "gemini-flash-lite"
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 1024
    search_mode: str = "hybrid"
    thinking: bool = False
    reranker_key: Optional[str] = None
    rerank_candidates: Optional[int] = None

class Citation(BaseModel):
    id: int
    page_title: str
    page_url: str
    section: str
    cited_text: str

class ImageResult(BaseModel):
    url: str
    alt_text: str
    section: str
    caption: str
    page_title: str

class ChatResponse(BaseModel):
    response: str
    citations: List[dict]
    images: List[dict]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest):
    logger.info(f"Received request: {body.message[:80]!r} (model: {body.model}, search: {body.search_mode}, reranker: {body.reranker_key})")
    try:
        # 1. Search DB based on search_mode
        search_results = search_engine.search(
            body.message,
            mode=body.search_mode,
            reranker_key=body.reranker_key,
            rerank_candidates=body.rerank_candidates,
        )
            
        logger.info(f"Retrieved {len(search_results)} search results.")

        # 2. Build Generator Config
        llm_info = LLM_MODELS.get(body.model)
        if not llm_info:
            raise HTTPException(status_code=400, detail=f"Model {body.model} not found in LLM_MODELS")
            
        if llm_info.backend == "ollama":
            base_url = "http://localhost:11434/v1"
            api_key = "ollama"
        elif llm_info.backend == "openrouter":
            from backend.config.settings import settings
            base_url = "https://openrouter.ai/api/v1"
            import os
            api_key = os.environ.get("OPENROUTER_API_KEY", settings.openrouter_api_key)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown backend for model: {llm_info.backend}")

        generator_config = GeneratorConfig(
            model=llm_info.model_id,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            base_url=base_url,
            api_key=api_key,
            thinking=body.thinking,
        )
        generator = AnswerGenerator(config=generator_config)
        
        # 3. Generate Answer
        # Pass conversation history appropriately (exclude the current message which is last)
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in body.history[:-1]]
        generated_answer = generator.generate(
            body.message, 
            search_results, 
            conversation_history=history_dicts,
            user_images=body.images
        )
        
        return ChatResponse(
            response=generated_answer.content,
            citations=generated_answer.citations,
            images=generated_answer.images
        )
        
    except Exception as e:
        logger.exception("Error during chat processing")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, body: ChatRequest):
    """SSE streaming variant of /chat. Sends tokens as they arrive."""
    logger.info(f"[stream] {body.message[:80]!r} (model: {body.model})")

    # 1. Search (runs synchronously, same as /chat)
    search_results = search_engine.search(
        body.message,
        mode=body.search_mode,
        reranker_key=body.reranker_key,
        rerank_candidates=body.rerank_candidates,
    )
    logger.info(f"[stream] {len(search_results)} search results")

    # 2. Build generator
    llm_info = LLM_MODELS.get(body.model)
    if not llm_info:
        raise HTTPException(status_code=400, detail=f"Model {body.model} not found")

    if llm_info.backend == "ollama":
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"
    elif llm_info.backend == "openrouter":
        from backend.config.settings import settings as _settings
        import os
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY", _settings.openrouter_api_key)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown backend: {llm_info.backend}")

    generator_config = GeneratorConfig(
        model=llm_info.model_id,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        base_url=base_url,
        api_key=api_key,
        thinking=body.thinking,
    )
    generator = AnswerGenerator(config=generator_config)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in body.history[:-1]]

    def event_generator():
        try:
            for event, data in generator.generate_stream(
                body.message,
                search_results,
                conversation_history=history_dicts,
                user_images=body.images,
            ):
                if event == "citations":
                    yield f"event: citations\ndata: {json.dumps(data)}\n\n"
                elif event == "token":
                    yield f"event: token\ndata: {json.dumps(data)}\n\n"
                elif event == "done":
                    yield f"event: done\ndata: {json.dumps(data)}\n\n"
                elif event == "error":
                    yield f"event: error\ndata: {json.dumps(data)}\n\n"
        except Exception as exc:
            logger.exception("[stream] generator error")
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("backend.api.server:app", host="127.0.0.1", port=8765, reload=True, reload_dirs=["backend"])
