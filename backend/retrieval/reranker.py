"""Reranker abstraction for second-stage passage reranking.

Supports local CrossEncoder rerankers and ZeroEntropy's hosted reranker API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests

from backend.config.settings import RERANKER_MODELS, settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankScore:
    index: int
    score: float


class BaseReranker:
    def rerank(self, query: str, documents: list[str], top_n: Optional[int] = None) -> list[RerankScore]:
        raise NotImplementedError


class LocalCrossEncoderReranker(BaseReranker):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            kwargs: dict = {}
            if self.model_id.startswith(("Qwen/", "zeroentropy/")):
                # These rerankers require remote-code aware loading.
                kwargs["automodel_args"] = {"trust_remote_code": True}
                kwargs["tokenizer_args"] = {"trust_remote_code": True}

            self._model = CrossEncoder(self.model_id, **kwargs)
            logger.info(f"Loaded reranker model: {self.model_id}")
        return self._model

    def rerank(self, query: str, documents: list[str], top_n: Optional[int] = None) -> list[RerankScore]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        order = np.argsort(-np.asarray(scores, dtype=float))
        if top_n is not None:
            order = order[:top_n]
        return [RerankScore(index=int(i), score=float(scores[i])) for i in order]


class ZeroEntropyReranker(BaseReranker):
    API_URL = "https://api.zeroentropy.dev/v1/models/rerank"

    def __init__(self, model_id: str, api_key: str):
        self.model_id = model_id
        self.api_key = api_key
        if not self.api_key:
            raise RuntimeError(
                "ZEROENTROPY_API_KEY not configured; required for zerank-2 reranking"
            )

    def rerank(self, query: str, documents: list[str], top_n: Optional[int] = None) -> list[RerankScore]:
        if not documents:
            return []
        response = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_id,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "latency": "fast",
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            RerankScore(index=int(item["index"]), score=float(item["relevance_score"]))
            for item in payload.get("results", [])
        ]


_RERANKER_CACHE: dict[str, BaseReranker] = {}


def get_reranker(model_key: Optional[str]) -> Optional[BaseReranker]:
    if not model_key:
        return None
    if model_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[model_key]

    info = RERANKER_MODELS.get(model_key)
    if not info:
        raise KeyError(f"Unknown reranker key: {model_key}. Available: {list(RERANKER_MODELS)}")

    if info.backend == "local":
        reranker: BaseReranker = LocalCrossEncoderReranker(info.model_id)
    elif info.backend == "zeroentropy":
        reranker = ZeroEntropyReranker(info.model_id, settings.zeroentropy_api_key)
    else:
        raise ValueError(f"Unsupported reranker backend: {info.backend}")

    _RERANKER_CACHE[model_key] = reranker
    return reranker