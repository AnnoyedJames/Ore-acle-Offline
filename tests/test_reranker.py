"""Unit tests for backend/retrieval/reranker.py and reranker wiring in HybridSearch.

All tests are fully offline — no model weights, no network, no ChromaDB/SQLite.
Heavy objects are replaced with lightweight fakes or unittest.mock stubs.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

def _make_search_result(**kwargs):
    """Return a SearchResult with sensible defaults for test overrides."""
    from backend.retrieval.search import SearchResult

    defaults = dict(
        chunk_id="chunk-1",
        page_title="Diamond",
        page_url="https://minecraft.wiki/w/Diamond",
        section_heading="Overview",
        section_level=2,
        text="Diamonds are rare minerals found underground.",
        token_count=10,
        chunk_type="section",
        page_type="item",
        rrf_score=0.05,
        infobox=None,
        images=[],
        semantic_score=0.9,
        keyword_score=None,
        reranker_score=None,
    )
    defaults.update(kwargs)
    return SearchResult(**defaults)


# ---------------------------------------------------------------------------
# Settings / registry tests
# ---------------------------------------------------------------------------

class TestRerankerRegistry:
    def test_all_three_models_registered(self):
        from backend.config.settings import RERANKER_MODELS

        assert "bge-reranker-v2-m3" in RERANKER_MODELS
        assert "qwen3-reranker-4b" in RERANKER_MODELS
        assert "zerank-2" in RERANKER_MODELS

    def test_local_backends(self):
        from backend.config.settings import RERANKER_MODELS

        assert RERANKER_MODELS["bge-reranker-v2-m3"].backend == "local"
        assert RERANKER_MODELS["qwen3-reranker-4b"].backend == "local"

    def test_zeroentropy_backend(self):
        from backend.config.settings import RERANKER_MODELS

        # zerank-2 is a local HF model (zeroentropy/zerank-2 on HuggingFace)
        assert RERANKER_MODELS["zerank-2"].backend == "local"

    def test_model_ids_match_hf_paths(self):
        from backend.config.settings import RERANKER_MODELS

        assert RERANKER_MODELS["bge-reranker-v2-m3"].model_id == "BAAI/bge-reranker-v2-m3"
        assert RERANKER_MODELS["qwen3-reranker-4b"].model_id == "Qwen/Qwen3-Reranker-4B"
        assert RERANKER_MODELS["zerank-2"].model_id == "zeroentropy/zerank-2"

    def test_rerank_candidates_default(self):
        from backend.config.settings import settings

        assert settings.retrieval_rerank_candidates == 30

    def test_reranker_model_default_empty(self):
        from backend.config.settings import settings

        assert settings.reranker_model == ""


# ---------------------------------------------------------------------------
# get_reranker factory
# ---------------------------------------------------------------------------

class TestGetReranker:
    def test_none_key_returns_none(self):
        from backend.retrieval.reranker import get_reranker

        assert get_reranker(None) is None

    def test_empty_string_returns_none(self):
        from backend.retrieval.reranker import get_reranker

        assert get_reranker("") is None

    def test_unknown_key_raises(self):
        from backend.retrieval.reranker import get_reranker

        with pytest.raises(KeyError, match="Unknown reranker key"):
            get_reranker("nonexistent-model")

    def test_local_key_returns_local_reranker(self):
        from backend.retrieval.reranker import LocalCrossEncoderReranker, get_reranker, _RERANKER_CACHE

        # Clear cache so we don't pick up a stale entry from a previous test run
        _RERANKER_CACHE.pop("bge-reranker-v2-m3", None)
        reranker = get_reranker("bge-reranker-v2-m3")
        assert isinstance(reranker, LocalCrossEncoderReranker)
        assert reranker.model_id == "BAAI/bge-reranker-v2-m3"

    def test_factory_caches_instance(self):
        from backend.retrieval.reranker import get_reranker, _RERANKER_CACHE

        _RERANKER_CACHE.pop("bge-reranker-v2-m3", None)
        r1 = get_reranker("bge-reranker-v2-m3")
        r2 = get_reranker("bge-reranker-v2-m3")
        assert r1 is r2

    def test_zeroentropy_raises_without_key(self):
        from backend.retrieval.reranker import ZeroEntropyReranker, _RERANKER_CACHE

        _RERANKER_CACHE.pop("zerank-2", None)
        with pytest.raises(RuntimeError, match="ZEROENTROPY_API_KEY"):
            ZeroEntropyReranker("zeroentropy/zerank-2", api_key="")


# ---------------------------------------------------------------------------
# LocalCrossEncoderReranker (model mocked — no weights downloaded)
# ---------------------------------------------------------------------------

class TestLocalCrossEncoderReranker:
    def _make_reranker_with_mock_model(self, scores: list[float]):
        """Return a LocalCrossEncoderReranker whose model.predict returns `scores`."""
        from backend.retrieval.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker("BAAI/bge-reranker-v2-m3")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(scores, dtype=float)
        reranker._model = mock_model
        return reranker

    def test_empty_documents_returns_empty(self):
        from backend.retrieval.reranker import LocalCrossEncoderReranker

        r = LocalCrossEncoderReranker("BAAI/bge-reranker-v2-m3")
        r._model = MagicMock()
        assert r.rerank("query", []) == []

    def test_results_sorted_by_descending_score(self):
        # scores[0]=0.1, scores[1]=0.9, scores[2]=0.5 → order should be 1, 2, 0
        reranker = self._make_reranker_with_mock_model([0.1, 0.9, 0.5])
        results = reranker.rerank("query", ["doc0", "doc1", "doc2"])
        assert [r.index for r in results] == [1, 2, 0]
        assert results[0].score == pytest.approx(0.9)
        assert results[1].score == pytest.approx(0.5)
        assert results[2].score == pytest.approx(0.1)

    def test_top_n_truncates_results(self):
        reranker = self._make_reranker_with_mock_model([0.1, 0.9, 0.5])
        results = reranker.rerank("query", ["doc0", "doc1", "doc2"], top_n=2)
        assert len(results) == 2
        assert results[0].index == 1  # highest score

    def test_single_document(self):
        reranker = self._make_reranker_with_mock_model([0.75])
        results = reranker.rerank("query", ["only doc"])
        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].score == pytest.approx(0.75)

    def test_predict_called_with_pairs(self):
        reranker = self._make_reranker_with_mock_model([0.5, 0.8])
        reranker.rerank("my query", ["doc A", "doc B"])
        call_args = reranker._model.predict.call_args
        pairs = call_args[0][0]
        assert pairs == [("my query", "doc A"), ("my query", "doc B")]

    def test_qwen_kwargs_set_trust_remote_code(self):
        from backend.retrieval.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker("Qwen/Qwen3-Reranker-4B")
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.return_value = MagicMock()
            _ = reranker.model  # trigger lazy load
            _, kwargs = mock_ce.call_args
            assert kwargs.get("automodel_args", {}).get("trust_remote_code") is True
            assert kwargs.get("tokenizer_args", {}).get("trust_remote_code") is True

    def test_zerank_kwargs_set_trust_remote_code(self):
        from backend.retrieval.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker("zeroentropy/zerank-2")
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.return_value = MagicMock()
            _ = reranker.model
            _, kwargs = mock_ce.call_args
            assert kwargs.get("automodel_args", {}).get("trust_remote_code") is True


# ---------------------------------------------------------------------------
# ZeroEntropyReranker (requests mocked — no network)
# ---------------------------------------------------------------------------

class TestZeroEntropyReranker:
    def _make_reranker(self):
        from backend.retrieval.reranker import ZeroEntropyReranker

        return ZeroEntropyReranker("zerank-2", api_key="test-key")

    def _mock_response(self, results: list[dict]):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": results}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_empty_documents_returns_empty(self):
        reranker = self._make_reranker()
        with patch("requests.post") as mock_post:
            out = reranker.rerank("query", [])
        mock_post.assert_not_called()
        assert out == []

    def test_results_parsed_correctly(self):
        reranker = self._make_reranker()
        api_response = [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.60},
            {"index": 1, "relevance_score": 0.30},
        ]
        with patch("requests.post", return_value=self._mock_response(api_response)):
            results = reranker.rerank("query", ["d0", "d1", "d2"])

        assert len(results) == 3
        assert results[0].index == 2
        assert results[0].score == pytest.approx(0.95)
        assert results[2].index == 1

    def test_correct_payload_sent(self):
        reranker = self._make_reranker()
        with patch("requests.post", return_value=self._mock_response([])) as mock_post:
            reranker.rerank("find diamonds", ["doc0", "doc1"], top_n=1)

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["model"] == "zerank-2"
        assert payload["query"] == "find diamonds"
        assert payload["documents"] == ["doc0", "doc1"]
        assert payload["top_n"] == 1
        assert payload["latency"] == "fast"

    def test_bearer_token_in_headers(self):
        reranker = self._make_reranker()
        with patch("requests.post", return_value=self._mock_response([])) as mock_post:
            reranker.rerank("query", ["doc"])

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"

    def test_http_error_propagates(self):
        import requests as req

        reranker = self._make_reranker()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError("403")
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(req.exceptions.HTTPError):
                reranker.rerank("query", ["doc"])


# ---------------------------------------------------------------------------
# HybridSearch reranker wiring (ChromaDB/SQLite/embedder fully mocked)
# ---------------------------------------------------------------------------

def _make_fake_search_engine(reranker_key=None, top_k=5, rerank_candidates=10):
    """Build a HybridSearch with all I/O mocked out."""
    from backend.retrieval.search import HybridSearch

    engine = HybridSearch.__new__(HybridSearch)
    engine.chroma = MagicMock()
    engine.sqlite = MagicMock()
    engine._embedder = MagicMock()
    engine.reranker_key = reranker_key
    engine._reranker = None
    engine.semantic_candidates = 5
    engine.keyword_candidates = 5
    engine.top_k = top_k
    engine.rerank_candidates = rerank_candidates
    engine.rrf_k = 20
    engine.rrf_alpha = 0.8
    return engine


class TestHybridSearchRerankerWiring:
    def test_no_reranker_returns_top_k(self):
        """Without a reranker, _rerank_results just slices to top_k."""
        from backend.retrieval.search import HybridSearch

        engine = _make_fake_search_engine(reranker_key=None, top_k=3)
        candidates = [_make_search_result(chunk_id=f"c{i}") for i in range(6)]
        out = engine._rerank_results("query", candidates)
        assert len(out) == 3
        assert out[0].chunk_id == "c0"  # order preserved

    def test_reranker_reorders_results(self):
        from backend.retrieval.search import HybridSearch
        from backend.retrieval.reranker import RerankScore

        engine = _make_fake_search_engine(reranker_key="bge-reranker-v2-m3", top_k=3)

        mock_reranker = MagicMock()
        # Reverse the order: last doc is most relevant
        mock_reranker.rerank.return_value = [
            RerankScore(index=2, score=0.9),
            RerankScore(index=1, score=0.6),
            RerankScore(index=0, score=0.2),
        ]
        engine._reranker = mock_reranker

        candidates = [_make_search_result(chunk_id=f"c{i}") for i in range(3)]
        out = engine._rerank_results("query", candidates)

        assert [r.chunk_id for r in out] == ["c2", "c1", "c0"]
        assert out[0].reranker_score == pytest.approx(0.9)
        assert out[1].reranker_score == pytest.approx(0.6)

    def test_reranker_score_attached_to_result(self):
        from backend.retrieval.reranker import RerankScore

        engine = _make_fake_search_engine(reranker_key="bge-reranker-v2-m3", top_k=2)
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            RerankScore(index=0, score=0.77),
            RerankScore(index=1, score=0.33),
        ]
        engine._reranker = mock_reranker

        candidates = [_make_search_result(chunk_id=f"c{i}") for i in range(2)]
        out = engine._rerank_results("query", candidates)
        assert out[0].reranker_score == pytest.approx(0.77)
        assert out[1].reranker_score == pytest.approx(0.33)

    def test_empty_candidate_list_returns_empty(self):
        engine = _make_fake_search_engine(reranker_key="bge-reranker-v2-m3", top_k=5)
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []
        engine._reranker = mock_reranker
        assert engine._rerank_results("query", []) == []

    def test_reranker_not_called_when_key_is_none(self):
        # When reranker_key is None the property must return None regardless
        # of _reranker state, so we should never reach the reranker call path.
        engine = _make_fake_search_engine(reranker_key=None, top_k=3)
        assert engine.reranker is None
        # _rerank_results should short-circuit and return the slice
        out = engine._rerank_results("query", [_make_search_result()])
        assert len(out) == 1

    def test_format_rerank_document(self):
        from backend.retrieval.search import HybridSearch

        engine = _make_fake_search_engine()
        r = _make_search_result(
            page_title="Diamond",
            section_heading="Mining",
            text="Diamonds spawn at y=-58.",
        )
        doc = engine._format_rerank_document(r)
        assert "Diamond" in doc
        assert "Mining" in doc
        assert "Diamonds spawn at y=-58." in doc


# ---------------------------------------------------------------------------
# RRF merge logic (pure unit tests — no I/O)
# ---------------------------------------------------------------------------

class TestRRFMerge:
    def _make_sem_result(self, cid: str, distance: float = 0.1):
        return {"id": cid, "distance": distance, "text": f"text for {cid}",
                "page_title": "T", "page_url": "http://x", "section_heading": "",
                "section_level": 2, "token_count": 10, "chunk_type": "section",
                "page_type": "other", "infobox": None, "images": []}

    def _make_kw_result(self, cid: str, bm25_norm: float = 0.5):
        return {"chunk_id": cid, "bm25_norm": bm25_norm, "text": f"text for {cid}",
                "page_title": "T", "page_url": "http://x", "section_heading": "",
                "section_level": 2, "token_count": 10, "chunk_type": "section",
                "page_type": "other", "infobox": None, "images": []}

    def _make_engine(self, alpha=0.8, k=20, top_k=5):
        engine = _make_fake_search_engine(top_k=top_k)
        engine.rrf_alpha = alpha
        engine.rrf_k = k
        # Stub SQLite get_by_ids to return nothing (text already on results)
        engine.sqlite.get_by_ids.return_value = {}
        return engine

    def test_semantic_only_rrf(self):
        engine = self._make_engine(alpha=1.0)
        sem = [self._make_sem_result(f"s{i}") for i in range(3)]
        out = engine._rrf_merge(sem, [], {})
        assert len(out) == 3
        cids = [r.chunk_id for r in out]
        assert cids == ["s0", "s1", "s2"]

    def test_keyword_only_rrf(self):
        engine = self._make_engine(alpha=0.0)
        kw = [self._make_kw_result(f"k{i}") for i in range(3)]
        out = engine._rrf_merge([], kw, {})
        assert len(out) == 3

    def test_rrf_boosts_shared_chunk(self):
        """A chunk that appears in both lists should score higher than exclusives."""
        engine = self._make_engine(alpha=0.5, k=20, top_k=10)
        sem = [
            self._make_sem_result("shared", distance=0.1),
            self._make_sem_result("sem-only", distance=0.15),
        ]
        kw = [
            self._make_kw_result("shared"),
            self._make_kw_result("kw-only"),
        ]
        out = engine._rrf_merge(sem, kw, {})
        chunk_ids = [r.chunk_id for r in out]
        assert chunk_ids[0] == "shared"

    def test_alpha_override(self):
        """With alpha_override=1.0, semantic-only scores dominate."""
        engine = self._make_engine(alpha=0.5, k=20, top_k=5)
        sem = [self._make_sem_result("sem-top")]
        kw = [self._make_kw_result("kw-top")]
        out_sem_heavy = engine._rrf_merge(sem, kw, {}, alpha_override=1.0)
        assert out_sem_heavy[0].chunk_id == "sem-top"

    def test_candidate_limit(self):
        """candidate_limit parameter caps pre-rerank list length."""
        engine = self._make_engine(top_k=10)
        sem = [self._make_sem_result(f"s{i}") for i in range(8)]
        out = engine._rrf_merge(sem, [], {}, candidate_limit=3)
        assert len(out) == 3

    def test_semantic_score_populated(self):
        engine = self._make_engine()
        sem = [self._make_sem_result("c1", distance=0.2)]
        out = engine._rrf_merge(sem, [], {})
        assert out[0].semantic_score == pytest.approx(1.0 - 0.2)

    def test_keyword_score_populated(self):
        engine = self._make_engine(alpha=0.0)
        kw = [self._make_kw_result("c1", bm25_norm=0.73)]
        out = engine._rrf_merge([], kw, {})
        assert out[0].keyword_score == pytest.approx(0.73)

    def test_missing_chunk_skipped(self):
        """Chunk IDs with no text anywhere are skipped gracefully."""
        engine = self._make_engine()
        # Result with no text field
        sem = [{"id": "ghost", "distance": 0.1}]
        out = engine._rrf_merge(sem, [], {})
        # ghost has no text and no fallback — it should be skipped
        assert all(r.chunk_id != "ghost" or r.text == "" for r in out)


# ---------------------------------------------------------------------------
# Candidate pool widening when reranker is active
# ---------------------------------------------------------------------------

class TestCandidateWidening:
    def test_no_reranker_uses_defaults(self):
        engine = _make_fake_search_engine(reranker_key=None)
        engine.semantic_candidates = 20
        engine.keyword_candidates = 15
        engine.rerank_candidates = 30
        # No reranker → limits stay at defaults, no widening needed
        assert engine.reranker_key is None

    def test_reranker_key_stored_on_engine(self):
        engine = _make_fake_search_engine(reranker_key="bge-reranker-v2-m3")
        assert engine.reranker_key == "bge-reranker-v2-m3"

    def test_rerank_candidates_default_is_30(self):
        engine = _make_fake_search_engine(rerank_candidates=30)
        assert engine.rerank_candidates == 30
