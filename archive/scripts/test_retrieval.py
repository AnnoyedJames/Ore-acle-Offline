#!/usr/bin/env python3
"""
Test Retrieval — verify the RAG pipeline end-to-end.

Runs sample queries through hybrid search + answer generation
and prints results to the console.

Usage:
    cd backend
    python -m scripts.test_retrieval
    python -m scripts.test_retrieval --query "How do I find diamonds?"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


SAMPLE_QUERIES = [
    "How do I find diamonds in Minecraft?",
    "What mobs spawn in the Nether?",
    "How does enchanting work?",
    "What is the best food source in Minecraft?",
    "How do I make a brewing stand?",
]


def test_search_only(query: str):
    """Test hybrid search without answer generation."""
    from retrieval.search import HybridSearch

    search = HybridSearch()
    results = search.search(query)

    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}")

    for i, r in enumerate(results):
        print(f"\n--- Result #{i + 1} (RRF: {r.rrf_score:.4f}) ---")
        print(f"  Page:    {r.page_title}")
        print(f"  Section: {r.section_heading}")
        print(f"  Type:    {r.page_type} / {r.chunk_type}")
        print(f"  Tokens:  {r.token_count}")
        if r.semantic_score is not None:
            print(f"  Semantic: {r.semantic_score:.4f}")
        if r.keyword_score is not None:
            print(f"  Keyword:  {r.keyword_score:.4f}")
        print(f"  Text:    {r.text[:200]}...")

    return results


def test_full_pipeline(query: str):
    """Test hybrid search + answer generation."""
    from retrieval.search import HybridSearch
    from retrieval.answer import AnswerGenerator

    # Search
    search = HybridSearch()
    results = search.search(query)

    if not results:
        print(f"\nNo results found for: {query}")
        return

    # Generate answer
    generator = AnswerGenerator()
    answer = generator.generate(query, results)

    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}")
    print(f"\nAnswer:\n{answer.content}")
    print(f"\nCitations ({len(answer.citations)}):")
    for c in answer.citations:
        print(f"  [{c['id']}] {c['page_title']} > {c['section']}")
    if answer.images:
        print(f"\nImages ({len(answer.images)}):")
        for img in answer.images:
            print(f"  - {img['alt_text']} ({img['page_title']})")
    print(f"\nUsage: {answer.usage}")


def main():
    parser = argparse.ArgumentParser(description="Test Ore-acle retrieval")
    parser.add_argument("--query", type=str, help="Custom query to test")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Only test search, skip answer generation",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all sample queries",
    )
    args = parser.parse_args()

    queries = [args.query] if args.query else SAMPLE_QUERIES if args.all else SAMPLE_QUERIES[:2]

    for query in queries:
        if args.search_only:
            test_search_only(query)
        else:
            test_full_pipeline(query)


if __name__ == "__main__":
    main()
