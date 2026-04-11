import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (lazy loaded or initialized here)
search_engine = HybridSearch()
# We instantiate AnswerGenerator per request to handle different configs

class Message(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[Message]
    model: str = "gemini-flash-lite"
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 1024
    search_mode: str = "hybrid"
    thinking: bool = False

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
async def chat(request: ChatRequest, req: Request):
    logger.info(f"Received request: {request.message[:80]!r} (model: {request.model}, search: {request.search_mode})")
    try:
        # 1. Search DB based on search_mode
        search_results = search_engine.search(request.message, mode=request.search_mode)
            
        logger.info(f"Retrieved {len(search_results)} search results.")

        # 2. Build Generator Config
        llm_info = LLM_MODELS.get(request.model)
        if not llm_info:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found in LLM_MODELS")
            
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
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            base_url=base_url,
            api_key=api_key,
            thinking=request.thinking,
        )
        generator = AnswerGenerator(config=generator_config)
        
        # 3. Generate Answer
        # Pass conversation history appropriately (exclude the current message which is last)
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history[:-1]]
        generated_answer = generator.generate(
            request.message, 
            search_results, 
            conversation_history=history_dicts
        )
        
        return ChatResponse(
            response=generated_answer.content,
            citations=generated_answer.citations,
            images=generated_answer.images
        )
        
    except Exception as e:
        logger.exception("Error during chat processing")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.api.server:app", host="127.0.0.1", port=8000, reload=True)
