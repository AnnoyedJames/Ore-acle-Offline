"""LangChain RecursiveCharacterTextSplitter-based chunker.

A simple baseline chunker for ablation comparison against the custom
section-aware chunker.  Splits each page's full text using LangChain's
``RecursiveCharacterTextSplitter`` with tiktoken token counting.

Produces the same ``Chunk`` dataclass used by the section-aware chunker
so the rest of the pipeline (embedder, ingestor, search) is unaffected.

Usage:
    from backend.preprocessing.langchain_chunker import LangChainChunker
    chunker = LangChainChunker()
    chunker.run()  # reads metadata.json → writes chunks.json
"""

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import tiktoken
from tqdm import tqdm

from backend.preprocessing.chunker import Chunk, ChunkerConfig

logger = logging.getLogger(__name__)


class LangChainChunker:
    """Baseline chunker using LangChain RecursiveCharacterTextSplitter."""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.enc = tiktoken.get_encoding(self.config.tiktoken_encoding)

    def _count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    @staticmethod
    def _make_chunk_id(page_title: str, index: int) -> str:
        raw = f"{page_title}|langchain|{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def run(self) -> list[dict]:
        """Read metadata.json, chunk every page, write chunks.json."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.config.tiktoken_encoding,
            chunk_size=self.config.max_tokens,
            chunk_overlap=self.config.overlap_tokens,
        )

        # Load page metadata
        logger.info(f"Loading metadata from {self.config.metadata_file}")
        with open(self.config.metadata_file, "r", encoding="utf-8") as f:
            pages = json.load(f)
        logger.info(f"Total pages: {len(pages)}")

        # Load page classifications if available
        classifications: dict[str, str] = {}
        if self.config.classified_pages_file.exists():
            with open(self.config.classified_pages_file, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    classifications[entry.get("title", "")] = entry.get("page_type", "other")

        all_chunks: list[Chunk] = []

        for page in tqdm(pages, desc="LangChain chunking"):
            title = page.get("title", "")
            url = page.get("url", "")
            page_type = classifications.get(title, "other")

            # Gather all section text into one document
            sections = page.get("sections", [])
            full_text = "\n\n".join(
                s.get("text", "") for s in sections if s.get("text", "").strip()
            )
            if not full_text.strip():
                continue

            # Split
            text_chunks = splitter.split_text(full_text)

            # Collect page-level images and infobox
            images = page.get("images", [])
            infobox = None
            for s in sections:
                if s.get("heading", "").lower() == "infobox":
                    infobox = s.get("infobox") or s.get("data")
                    break

            for idx, chunk_text in enumerate(text_chunks):
                chunk = Chunk(
                    chunk_id=self._make_chunk_id(title, idx),
                    page_title=title,
                    page_url=url,
                    section_heading="",
                    section_level=2,
                    text=chunk_text,
                    token_count=self._count_tokens(chunk_text),
                    chunk_type="section",
                    page_type=page_type,
                    infobox=infobox if idx == 0 else None,
                    images=images if idx == 0 else [],
                )
                all_chunks.append(chunk)

        # Write output
        output = [asdict(c) for c in all_chunks]
        with open(self.config.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=1)

        logger.info(f"LangChain chunker produced {len(all_chunks)} chunks → {self.config.output_file}")
        return output


if __name__ == "__main__":
    chunker = LangChainChunker()
    chunker.run()
