"""Reranker abstraction for second-stage passage reranking.

Supports local CrossEncoder rerankers and ZeroEntropy's hosted reranker API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests
import torch

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
                kwargs["trust_remote_code"] = True

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


class LocalQwenReranker(BaseReranker):
    """Qwen-style reranker that scores yes/no logits from a causal LM.

    Qwen3 and zerank-2 expose reranking via a generative yes/no head rather than
    a standard sequence-classification CrossEncoder head. This follows the model
    card inference path to avoid loading a randomly initialized classifier head.
    """

    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    SYSTEM_PREFIX = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the Query '
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
        '<|im_end|>\n<|im_start|>user\n'
    )
    ASSISTANT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def __init__(
        self,
        model_id: str,
        instruction: Optional[str] = None,
        batch_size: int = 2,
        quantization: Optional[str] = None,
    ):
        self.model_id = model_id
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.batch_size = batch_size
        self.quantization = quantization
        self.max_length = 8192
        self._model = None
        self._tokenizer = None
        self._prefix_tokens: Optional[list[int]] = None
        self._suffix_tokens: Optional[list[int]] = None
        self._token_true_id: Optional[int] = None
        self._token_false_id: Optional[int] = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                padding_side="left",
                trust_remote_code=True,
                local_files_only=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            kwargs: dict = dict(
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype="auto",
                device_map="auto",
            )
            if self.quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                kwargs.pop("torch_dtype", None)
            elif self.quantization == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                kwargs.pop("torch_dtype", None)

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **kwargs,
            ).eval()
            logger.info(
                f"Loaded reranker model: {self.model_id}"
                + (f" ({self.quantization})" if self.quantization else "")
            )
        return self._model

    @property
    def prefix_tokens(self) -> list[int]:
        if self._prefix_tokens is None:
            self._prefix_tokens = self.tokenizer.encode(
                self.SYSTEM_PREFIX,
                add_special_tokens=False,
            )
        return self._prefix_tokens

    @property
    def suffix_tokens(self) -> list[int]:
        if self._suffix_tokens is None:
            self._suffix_tokens = self.tokenizer.encode(
                self.ASSISTANT_SUFFIX,
                add_special_tokens=False,
            )
        return self._suffix_tokens

    @property
    def token_true_id(self) -> int:
        if self._token_true_id is None:
            self._token_true_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        return self._token_true_id

    @property
    def token_false_id(self) -> int:
        if self._token_false_id is None:
            self._token_false_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        return self._token_false_id

    def _format_instruction(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _model_device(self):
        return next(self.model.parameters()).device

    def _prepare_inputs(self, query: str, documents: list[str]):
        formatted = [self._format_instruction(query, document) for document in documents]
        max_text_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = self.tokenizer(
            formatted,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_text_length,
        )
        for index, token_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = self.prefix_tokens + token_ids + self.suffix_tokens

        padded = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        device = self._model_device()
        return {key: value.to(device) for key, value in padded.items()}

    @torch.inference_mode()
    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        model_inputs = self._prepare_inputs(query, documents)
        batch_logits = self.model(**model_inputs).logits[:, -1, :]
        false_logits = batch_logits[:, self.token_false_id]
        true_logits = batch_logits[:, self.token_true_id]
        yes_no_logits = torch.stack([false_logits, true_logits], dim=1)
        probabilities = torch.nn.functional.softmax(yes_no_logits, dim=1)[:, 1]
        return probabilities.detach().float().cpu().tolist()

    def rerank(self, query: str, documents: list[str], top_n: Optional[int] = None) -> list[RerankScore]:
        if not documents:
            return []

        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start:start + self.batch_size]
            scores.extend(self._score_batch(query, batch))

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
        if info.model_id.startswith(("Qwen/", "zeroentropy/")):
            reranker = LocalQwenReranker(info.model_id, quantization=info.quantization)
        else:
            reranker = LocalCrossEncoderReranker(info.model_id)
    elif info.backend == "zeroentropy":
        reranker = ZeroEntropyReranker(info.model_id, settings.zeroentropy_api_key)
    else:
        raise ValueError(f"Unsupported reranker backend: {info.backend}")

    _RERANKER_CACHE[model_key] = reranker
    return reranker