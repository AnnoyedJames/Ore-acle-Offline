"""LangChain RecursiveCharacterTextSplitter-based chunker.

A simple baseline chunker for ablation comparison against the custom
section-aware chunker.  Splits each page's full text using LangChain's
``RecursiveCharacterTextSplitter`` with tiktoken token counting.

Produces the same ``Chunk`` dataclass used by the section-aware chunker
so the rest of the pipeline (embedder, ingestor, search) is unaffected.

Image binding strategy: track each section's character-offset range in the
concatenated full_text, then for each LangChain chunk, assign images from all
sections whose text overlaps with that chunk's character range.  This is more
faithful to the section-aware chunker than the naïve "all images on chunk 0"
approach, while staying within the spirit of a simple recursive splitter.

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

_SECTION_SEP = "\n\n"  # separator used when joining sections into full_text


_DEFAULT_LANGCHAIN_OUTPUT = Path("data/processed/chunks_langchain.json")


class LangChainChunker:
    """Baseline chunker using LangChain RecursiveCharacterTextSplitter."""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        # Override default output path to avoid clobbering the section-aware chunks
        if self.config.output_file == Path("data/processed/chunks.json"):
            self.config.output_file = _DEFAULT_LANGCHAIN_OUTPUT
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.enc = tiktoken.get_encoding(self.config.tiktoken_encoding)

    def _count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    @staticmethod
    def _make_chunk_id(page_title: str, index: int) -> str:
        raw = f"{page_title}|langchain|{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _build_full_text_with_offsets(
        sections: list[dict],
    ) -> tuple[str, list[dict]]:
        """
        Join non-empty section texts with ``\\n\\n`` separators and record the
        character-offset range [start, end) for each section in the result.

        Returns:
            full_text: the concatenated string
            offsets:   list of dicts with keys heading, start, end, images
        """
        parts: list[str] = []
        offsets: list[dict] = []
        cursor = 0
        for sec in sections:
            text = sec.get("text", "")
            if not text.strip():
                continue
            start = cursor
            end = cursor + len(text)
            offsets.append(
                {
                    "heading": sec.get("heading", ""),
                    "start": start,
                    "end": end,
                }
            )
            parts.append(text)
            cursor = end + len(_SECTION_SEP)  # account for the joining separator

        full_text = _SECTION_SEP.join(parts)
        return full_text, offsets

    def _assign_images_to_chunks(
        self,
        text_chunks: list[str],
        full_text: str,
        offsets: list[dict],
        page_images: list[dict],
    ) -> list[list[dict]]:
        """
        For each text chunk, determine which sections it overlaps with by
        locating the chunk's character range in full_text, then return the
        images belonging to those sections.

        Images with no ``section`` tag (or whose tag doesn't match any known
        section heading) are assigned to the first chunk only, mirroring the
        infobox / lead-image convention used by the section-aware chunker.
        """
        # Build a lookup: normalised heading → list[dict]
        heading_images: dict[str, list[dict]] = {}
        untagged_images: list[dict] = []

        known_headings = {sec["heading"].strip().lower() for sec in offsets}

        for img in page_images:
            sec_tag = img.get("section", "").strip()
            if sec_tag and sec_tag.lower() in known_headings:
                key = sec_tag.lower()
                heading_images.setdefault(key, []).append(img)
            else:
                untagged_images.append(img)

        # Locate each chunk inside full_text (search forward to handle overlaps)
        chunk_images: list[list[dict]] = []
        search_from = 0

        for idx, chunk_text in enumerate(text_chunks):
            pos = full_text.find(chunk_text, search_from)
            if pos == -1:
                # Fallback: scan from beginning (handles rare edge cases)
                pos = full_text.find(chunk_text)

            if pos == -1:
                # Cannot locate chunk – assign no images (safe default)
                chunk_images.append(untagged_images if idx == 0 else [])
                continue

            chunk_start = pos
            chunk_end = pos + len(chunk_text)
            # Advance forward cursor minus overlap to avoid skipping chunks
            search_from = max(search_from, chunk_end - self.config.overlap_tokens * 4)

            # Find sections that overlap [chunk_start, chunk_end)
            matched: list[dict] = []
            for sec in offsets:
                if sec["end"] > chunk_start and sec["start"] < chunk_end:
                    imgs = heading_images.get(sec["heading"].strip().lower(), [])
                    for img in imgs:
                        if img not in matched:
                            matched.append(img)

            # Untagged images go to the first chunk only
            if idx == 0:
                for img in untagged_images:
                    if img not in matched:
                        matched.append(img)

            chunk_images.append(matched)

        return chunk_images

    def run(self) -> list[dict]:
        """Read metadata.json, chunk every page, write chunks.json."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.config.tiktoken_encoding,
            chunk_size=self.config.max_tokens,
            chunk_overlap=self.config.overlap_tokens,
        )

        # Load page metadata (metadata.json wraps pages in {"pages": [...]})
        logger.info(f"Loading metadata from {self.config.metadata_file}")
        with open(self.config.metadata_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        pages: list[dict] = raw.get("pages", raw) if isinstance(raw, dict) else raw
        logger.info(f"Total pages: {len(pages)}")

        # Load page classifications — classified_pages.json is {"pages": {title: {...}}}
        classifications: dict[str, str] = {}
        if self.config.classified_pages_file.exists():
            with open(self.config.classified_pages_file, "r", encoding="utf-8") as f:
                cls_raw = json.load(f)
            cls_map: dict = cls_raw.get("pages", cls_raw) if isinstance(cls_raw, dict) else {}
            # Support both {title: {page_type: ...}} and [{title, page_type}] formats
            if isinstance(cls_map, dict):
                for title, info in cls_map.items():
                    if isinstance(info, dict):
                        classifications[title] = info.get("page_type", "other")
                    else:
                        classifications[title] = "other"
            elif isinstance(cls_map, list):
                for entry in cls_map:
                    classifications[entry.get("title", "")] = entry.get("page_type", "other")

        all_chunks: list[Chunk] = []

        for page in tqdm(pages, desc="LangChain chunking"):
            title = page.get("title", "")
            url = page.get("url", "")
            page_type = classifications.get(title, "other")

            sections = page.get("sections", [])
            full_text, offsets = self._build_full_text_with_offsets(sections)
            if not full_text.strip():
                continue

            # Split the concatenated text
            text_chunks = splitter.split_text(full_text)

            # Collect page-level images and infobox
            page_images: list[dict] = page.get("images", [])
            infobox = None
            for s in sections:
                if s.get("heading", "").lower() == "infobox":
                    infobox = s.get("infobox") or s.get("data")
                    break

            # Assign images to each chunk based on section-offset overlap
            chunk_image_lists = self._assign_images_to_chunks(
                text_chunks, full_text, offsets, page_images
            )

            for idx, (chunk_text, chunk_imgs) in enumerate(
                zip(text_chunks, chunk_image_lists)
            ):
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
                    images=chunk_imgs,
                )
                all_chunks.append(chunk)

        # Write output
        output = [asdict(c) for c in all_chunks]
        with open(self.config.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=1)

        logger.info(
            f"LangChain chunker produced {len(all_chunks)} chunks → {self.config.output_file}"
        )
        return output


if __name__ == "__main__":
    chunker = LangChainChunker()
    chunker.run()
