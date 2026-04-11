"""LLM Client — unified interface for Ollama (local) and OpenRouter (API).

Routes requests based on the ``backend`` field in :data:`LLM_MODELS`.
Both backends use the OpenAI-compatible chat completions format.

Usage:
    from backend.retrieval.llm_client import get_llm_client
    client = get_llm_client("gemini-flash-lite")
    answer = client.generate("What is a creeper?", context_chunks)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from backend.config.settings import LLM_MODELS, LLMModelInfo
from backend.retrieval.answer import SYSTEM_PROMPT as EVAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Ollama default endpoint (OpenAI-compatible)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient:
    """Wraps an OpenAI-compatible client for a specific model.

    Automatically chooses the right base URL based on ``LLMModelInfo.backend``.
    """

    def __init__(self, model_key: str):
        info = LLM_MODELS.get(model_key)
        if info is None:
            raise ValueError(
                f"Unknown LLM '{model_key}'. Available: {list(LLM_MODELS)}"
            )
        self.model_key = model_key
        self.info = info
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        if self.info.backend == "ollama":
            self._client = OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama",  # Ollama ignores this but the SDK requires it
            )
        elif self.info.backend == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not set. Required for model "
                    f"'{self.model_key}' (backend=openrouter)."
                )
            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unknown backend: {self.info.backend}")

        logger.info(
            f"LLM client initialised: {self.info.label} "
            f"({self.info.backend} → {self.info.model_id})"
        )
        return self._client

    def generate(
        self,
        query: str,
        context: str,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> LLMResponse:
        """Generate an answer given a query and pre-formatted context string."""
        client = self._get_client()

        user_message = (
            f"Sources:\n{context}\n\n"
            f"Question: {query}"
        )

        response = client.chat.completions.create(
            model=self.info.model_id,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content.strip(),
            model=self.info.model_id,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )


def get_llm_client(model_key: str) -> LLMClient:
    """Factory — return an :class:`LLMClient` for *model_key*."""
    return LLMClient(model_key)
