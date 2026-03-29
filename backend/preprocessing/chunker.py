"""
Section-aware Chunker — splits cleaned pages into retrieval-sized chunks.

Each chunk carries full provenance metadata for NotebookLM-style citations:
page title, URL, section heading, section level, page type, categories,
related pages (interlinks), infobox data, and associated images.

Chunking strategy:
  1. Each section is a natural chunk boundary
  2. If a section fits within max_tokens → single chunk
  3. If a section exceeds max_tokens → split on paragraph boundaries
     with configurable token overlap
  4. Infobox gets its own chunk if present
  5. Tables get their own chunks (can be large in Markdown format)

Token counting uses tiktoken (cl100k_base, same as OpenAI / many models).

Usage:
    python -m preprocessing.chunker
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import tiktoken
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to import nltk for better sentence tokenization, fall back to regex
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
    USE_NLTK = True
except ImportError:
    USE_NLTK = False
    logger.warning("nltk not available, using regex sentence splitter")


@dataclass
class ChunkerConfig:
    """Configuration for the chunker."""
    metadata_file: Path = field(
        default_factory=lambda: Path("data/processed/metadata.json")
    )
    interlinks_file: Path = field(
        default_factory=lambda: Path("data/processed/interlinks.json")
    )
    classified_pages_file: Path = field(
        default_factory=lambda: Path("data/processed/classified_pages.json")
    )
    output_file: Path = field(
        default_factory=lambda: Path("data/processed/chunks.json")
    )
    max_tokens: int = 512
    overlap_tokens: int = 50
    # Minimum chunk size — skip near-empty sections
    min_tokens: int = 10
    # Chunks below this threshold get merged with adjacent same-page chunks
    merge_threshold: int = 50
    # Encoding for token counting
    tiktoken_encoding: str = "cl100k_base"


@dataclass
class Chunk:
    """A single retrieval chunk with full provenance metadata."""
    chunk_id: str
    page_title: str
    page_url: str
    section_heading: str
    section_level: int
    text: str
    token_count: int
    chunk_type: str  # "section", "infobox", "table"
    section_type: str = "content"  # "content", "history", "legacy", "future", "navigation"
    # Metadata for filtering / re-ranking
    page_type: str = "other"
    categories: list[str] = field(default_factory=list)
    related_pages: list[str] = field(default_factory=list)
    infobox: Optional[dict] = None
    images: list[dict] = field(default_factory=list)


class Chunker:
    """
    Section-aware chunker for Minecraft wiki pages.

    Reads metadata.json (text cleaner output), interlinks.json, and
    classified_pages.json, then produces chunks.json with full metadata.
    """

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.enc = tiktoken.get_encoding(self.config.tiktoken_encoding)

    def _count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def _make_chunk_id(self, page_title: str, section: str, index: int) -> str:
        """Deterministic chunk ID from page + section + sub-index."""
        raw = f"{page_title}|{section}|{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _is_list_item(self, line: str) -> bool:
        """Check if a line is a list item (bullet or numbered)."""
        return bool(re.match(r'^\s*[-*]\s+', line) or re.match(r'^\s*\d+\.\s+', line))

    def _extract_list_blocks(self, text: str) -> list[tuple[str, bool]]:
        """
        Split text into (block, is_list) tuples.
        Groups consecutive list items together as atomic units.
        """
        lines = text.split('\n')
        blocks = []
        current_block = []
        in_list = False

        for line in lines:
            line_is_list = self._is_list_item(line)

            if line_is_list:
                if not in_list and current_block:
                    # Flush non-list block
                    blocks.append(('\n'.join(current_block), False))
                    current_block = []
                current_block.append(line)
                in_list = True
            else:
                if in_list and current_block:
                    # Flush list block
                    blocks.append(('\n'.join(current_block), True))
                    current_block = []
                if line.strip():  # Only add non-empty lines
                    current_block.append(line)
                in_list = False

        if current_block:
            blocks.append(('\n'.join(current_block), in_list))

        return blocks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using nltk or fallback regex."""
        if USE_NLTK:
            return sent_tokenize(text)
        else:
            # Fallback regex
            parts = re.split(r'(?<=[.!?])\s+', text)
            return [p.strip() for p in parts if p.strip()]

    def _split_text_structure_aware(
        self, text: str, max_tokens: int, overlap_tokens: int
    ) -> list[str]:
        """
        Structure-aware splitting that preserves lists and uses sentence boundaries.
        
        Strategy:
        1. Detect and preserve list blocks as atomic units
        2. Split non-list blocks at sentence boundaries
        3. Use complete-sentence overlap (not mid-sentence token counts)
        4. Keep chunks under max_tokens
        5. Force-split any single unit (sentence/block) that exceeds max_tokens
        """
        # First, separate lists from prose
        blocks = self._extract_list_blocks(text)
        if not blocks:
            # Fallback for empty or single block
            if not text.strip():
                return []
            blocks = [(text, False)]

        chunks = []
        current_sentences: list[str] = []
        current_tokens = 0

        for block_text, is_list in blocks:
            if not block_text.strip():
                continue

            # If it's a list, use existing list splitting logic
            if is_list:
                list_lines = block_text.split('\n')
                for line in list_lines:
                    line_tokens = self._count_tokens(line)
                    
                    # If single line is too huge, force split it
                    if line_tokens > max_tokens:
                        if current_sentences:
                            chunks.append(' '.join(current_sentences))
                            current_sentences = []
                            current_tokens = 0
                        chunks.extend(self._force_split_text(line, max_tokens))
                        continue

                    # If adding line exceeds budget, flush
                    if current_tokens + line_tokens > max_tokens:
                        if current_sentences:
                            chunks.append(' '.join(current_sentences))
                        
                        # Start new chunk with overlaps handled? Lists usually don't need prose overlap.
                        # Simple flush for lists.
                        current_sentences = []
                        current_tokens = 0

                    current_sentences.append(line)
                    current_tokens += line_tokens
                continue

            # Prose block — split by sentences
            sentences = self._split_into_sentences(block_text)
            for sent in sentences:
                sent_tokens = self._count_tokens(sent)

                # If sentence itself is huge, force split it
                if sent_tokens > max_tokens:
                    if current_sentences:
                        chunks.append(' '.join(current_sentences))
                        current_sentences = []
                        current_tokens = 0
                    
                    # Force split the giant sentence
                    sub_chunks = self._force_split_text(sent, max_tokens)
                    chunks.extend(sub_chunks)
                    continue

                if current_tokens + sent_tokens > max_tokens and current_sentences:
                    # Flush current chunk
                    chunks.append(' '.join(current_sentences))
                    
                    # Overlap: keep last N complete sentences
                    overlap_sents = []
                    overlap_tok = 0
                    for s in reversed(current_sentences):
                        stok = self._count_tokens(s)
                        if overlap_tok + stok > overlap_tokens:
                            break
                        overlap_sents.insert(0, s)
                        overlap_tok += stok
                    
                    current_sentences = overlap_sents
                    current_tokens = overlap_tok

                current_sentences.append(sent)
                current_tokens += sent_tokens

        if current_sentences:
            chunks.append(' '.join(current_sentences))

        return chunks

    def _force_split_text(self, text: str, max_tokens: int) -> list[str]:
        """Hard split text that exceeds max_tokens, breaking on words or chars."""
        words = text.split(' ')
        chunks = []
        current_chunk = []
        current_toks = 0
        
        for word in words:
            wt = self._count_tokens(word)
            
            # Massive single word? (unlikely in wiki text but possible in code blocks)
            # If so, force split by arbitrary chunks
            if wt > max_tokens:
                # Flush existing buffer first
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_toks = 0
                
                # Split the massive word itself by character count approximation
                # or just hard chunk by tokens (slower but safer)
                # Simple fallback: split by 1000 chars roughly equiv to 250 tokens
                # Let's just append it as is if we can't split it easily, 
                # OR return it as a single (still too big) chunk to avoid data loss.
                # Better: Chunk it.
                chunks.append(word) # Todo: implement char-level splitting if needed
                continue

            if current_toks + wt > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_toks = 0
            
            current_chunk.append(word)
            current_toks += wt
            
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks


    def _chunk_section(
        self,
        text: str,
        page_title: str,
        page_url: str,
        section_heading: str,
        section_level: int,
        chunk_type: str,
        page_type: str,
        categories: list[str],
        related_pages: list[str],
        infobox: Optional[dict],
        images: list[dict],
        section_type: str = "content",
    ) -> list[Chunk]:
        """Create chunk(s) from a single section text."""
        text = text.strip()
        if not text:
            return []

        token_count = self._count_tokens(text)
        if token_count < self.config.min_tokens:
            return []

        # Fits in one chunk
        if token_count <= self.config.max_tokens:
            chunk_id = self._make_chunk_id(
                page_title, 
                section_heading + (" (history)" if section_type == "history" else ""),
                0
            )
            return [Chunk(
                chunk_id=chunk_id,
                page_title=page_title,
                page_url=page_url,
                section_heading=section_heading,
                section_level=section_level,
                text=text,
                token_count=token_count,
                chunk_type=chunk_type,
                section_type=section_type,
                page_type=page_type,
                categories=categories,
                related_pages=related_pages,
                infobox=infobox,
                images=images,
            )]

        # Need to split with structure-aware strategy
        sub_texts = self._split_text_structure_aware(
            text, self.config.max_tokens, self.config.overlap_tokens
        )
        chunks = []
        for i, sub_text in enumerate(sub_texts):
            chunk_id = self._make_chunk_id(
                page_title, 
                section_heading + (" (history)" if section_type == "history" else ""),
                i
            )
            chunks.append(Chunk(
                chunk_id=chunk_id,
                page_title=page_title,
                page_url=page_url,
                section_heading=section_heading,
                section_level=section_level,
                text=sub_text,
                token_count=self._count_tokens(sub_text),
                chunk_type=chunk_type,
                section_type=section_type,
                page_type=page_type,
                categories=categories,
                related_pages=related_pages,
                infobox=infobox,
                images=images,
            ))

        return chunks

    def _split_table_by_rows(self, table_md: str) -> list[str]:
        """
        Split a large Markdown table into smaller chunks by token budget.

        Always includes the header row in each chunk. Uses max_tokens to
        determine how many rows fit per chunk, handling both wide tables
        (few rows, many columns) and tall tables (many rows, few columns).
        """
        lines = table_md.strip().split('\n')

        # Parse header, separator, and data rows
        header = None
        separator = None
        rows = []
        for line in lines:
            if header is None:
                header = line
            elif separator is None:
                separator = line
            else:
                rows.append(line)

        if not rows or header is None or separator is None:
            return [table_md]

        # Token budget per chunk = max_tokens minus header overhead
        header_block = f"{header}\n{separator}"
        header_tokens = self._count_tokens(header_block)
        row_budget = self.config.max_tokens - header_tokens

        if row_budget <= 0:
            # Header alone exceeds budget.
            # Strategy: Ignore budget and force 1 row per chunk to at least break it up.
            row_budget = 0

        chunks = []
        current_rows: list[str] = []
        current_tokens = 0

        for row in lines[2:]:  # processing rows skipping header/sep
            row_tokens = self._count_tokens(row)
            
            # If we have rows pending, and adding this one exceeds budget...
            # Special case: If row_budget <= 0 (huge header), we MUST flush after every row
            # unless current_rows is empty.
            if current_rows and (row_budget <= 0 or current_tokens + row_tokens > row_budget):
                chunk = '\n'.join([header, separator] + current_rows)
                chunks.append(chunk)
                current_rows = []
                current_tokens = 0

            current_rows.append(row)
            current_tokens += row_tokens

        # Flush remaining rows
        if current_rows:
            chunk = '\n'.join([header, separator] + current_rows)
            chunks.append(chunk)

        return chunks if chunks else [table_md]

    def chunk_page(
        self,
        page: dict,
        page_type: str,
        categories: list[str],
        related_pages: list[str],
    ) -> list[Chunk]:
        """Chunk a single ProcessedPage dict into retrieval chunks."""
        title = page.get("title", "")
        url = page.get("url", "")
        infobox = page.get("infobox")
        page_images = page.get("images", [])
        sections = page.get("sections", [])
        tables = page.get("tables", [])

        all_chunks: list[Chunk] = []

        # 1. Infobox as its own chunk (if present and non-empty)
        if infobox and isinstance(infobox, dict) and infobox:
            infobox_text = "\n".join(f"{k}: {v}" for k, v in infobox.items() if v)
            if infobox_text.strip():
                all_chunks.extend(self._chunk_section(
                    text=infobox_text,
                    page_title=title,
                    page_url=url,
                    section_heading="Infobox",
                    section_level=0,
                    chunk_type="infobox",
                    page_type=page_type,
                    categories=categories,
                    related_pages=related_pages,
                    infobox=infobox,
                    images=[img for img in page_images if img.get("context_type") == "infobox"],
                ))

        # 2. Sections
        for section in sections:
            heading = section.get("heading", "Introduction")
            level = section.get("level", 2)
            text = section.get("text", "")
            section_type = section.get("section_type", "content")

            # Find images relevant to this section
            section_images = [
                img for img in page_images
                if img.get("section", "") == heading
            ]

            all_chunks.extend(self._chunk_section(
                text=text,
                page_title=title,
                page_url=url,
                section_heading=heading,
                section_level=level,
                chunk_type="section",
                page_type=page_type,
                categories=categories,
                related_pages=related_pages,
                infobox=None,
                images=section_images,
                section_type=section_type,
            ))

        # 3. Tables: skipped because they are now integrated into sections text
        # for i, table_md in enumerate(tables): ...


        # 4. Merge small adjacent chunks from this page
        all_chunks = self._merge_small_chunks(all_chunks)

        return all_chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Merge small adjacent chunks from the same page.

        Combines consecutive chunks that are both below merge_threshold into
        a single chunk, as long as the merged result stays under max_tokens.
        This is futureproof: works for any page regardless of content type.

        Rules:
        - Only merges adjacent chunks of the same chunk_type
        - Merged chunk inherits the section_heading of the first chunk,
          with " + {other_heading}" appended if headings differ
        - Images from both chunks are combined
        - Token count is recomputed after merge
        """
        if len(chunks) <= 1:
            return chunks

        threshold = self.config.merge_threshold
        merged: list[Chunk] = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If current chunk is already big enough, keep it as-is
            if current.token_count >= threshold:
                merged.append(current)
                i += 1
                continue

            # Try to merge with subsequent small chunks of the same type
            combined_text = current.text
            combined_tokens = current.token_count
            combined_images = list(current.images)
            combined_heading = current.section_heading
            j = i + 1

            while j < len(chunks):
                nxt = chunks[j]
                # Only merge if the next chunk is also small and same type
                if nxt.token_count >= threshold or nxt.chunk_type != current.chunk_type:
                    break
                # Check the merged result won't exceed max_tokens
                candidate_tokens = combined_tokens + nxt.token_count
                if candidate_tokens > self.config.max_tokens:
                    break

                # Perform merge
                separator = "\n" if current.chunk_type == "table" else "\n\n"
                combined_text = combined_text + separator + nxt.text
                combined_tokens = self._count_tokens(combined_text)
                combined_images.extend(nxt.images)
                if nxt.section_heading != current.section_heading:
                    if nxt.section_heading not in combined_heading:
                        combined_heading = f"{combined_heading} + {nxt.section_heading}"
                j += 1

            # Create the merged chunk
            merged_id = self._make_chunk_id(
                current.page_title, combined_heading, len(merged)
            )
            merged.append(Chunk(
                chunk_id=merged_id,
                page_title=current.page_title,
                page_url=current.page_url,
                section_heading=combined_heading,
                section_level=current.section_level,
                text=combined_text,
                token_count=combined_tokens,
                chunk_type=current.chunk_type,
                page_type=current.page_type,
                categories=current.categories,
                related_pages=current.related_pages,
                infobox=current.infobox,
                images=combined_images,
            ))
            i = j

        return merged

    def chunk_all(self) -> list[dict]:
        """Chunk all pages from metadata.json using interlinks and classifications."""
        # Load inputs
        logger.info(f"Loading metadata from {self.config.metadata_file}")
        with open(self.config.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        pages = metadata.get("pages", [])
        logger.info(f"Pages to chunk: {len(pages)}")

        # Load interlinks (optional)
        interlinks: dict[str, list[str]] = {}
        if self.config.interlinks_file.exists():
            logger.info(f"Loading interlinks from {self.config.interlinks_file}")
            with open(self.config.interlinks_file, "r", encoding="utf-8") as f:
                il_data = json.load(f)
            interlinks = il_data.get("graph", {})
            logger.info(f"Interlinks loaded for {len(interlinks)} pages")

        # Load classifications (optional)
        classifications: dict[str, dict] = {}
        if self.config.classified_pages_file.exists():
            logger.info(f"Loading classifications from {self.config.classified_pages_file}")
            with open(self.config.classified_pages_file, "r", encoding="utf-8") as f:
                cls_data = json.load(f)
            classifications = cls_data.get("pages", {})
            logger.info(f"Classifications loaded for {len(classifications)} pages")

        # Process pages
        all_chunks: list[Chunk] = []
        skipped = 0

        for page in tqdm(pages, desc="Chunking pages"):
            title = page.get("title", "")

            # Get classification data
            cls = classifications.get(title, {})
            page_type = cls.get("page_type", "other")
            categories = cls.get("semantic_categories", page.get("categories", []))

            # Get related pages from interlinks
            related = interlinks.get(title, [])

            page_chunks = self.chunk_page(page, page_type, categories, related)
            if not page_chunks:
                skipped += 1
            all_chunks.extend(page_chunks)

        logger.info(f"Total chunks: {len(all_chunks)}")
        logger.info(f"Pages skipped (no content): {skipped}")

        # Token stats
        token_counts = [c.token_count for c in all_chunks]
        if token_counts:
            avg = sum(token_counts) / len(token_counts)
            logger.info(f"Avg tokens/chunk: {avg:.0f}")
            logger.info(f"Min: {min(token_counts)}, Max: {max(token_counts)}")

        # Chunk type distribution
        type_dist: dict[str, int] = {}
        for c in all_chunks:
            type_dist[c.chunk_type] = type_dist.get(c.chunk_type, 0) + 1
        logger.info(f"Chunk types: {type_dist}")

        # Save
        output = [asdict(c) for c in all_chunks]
        with open(self.config.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(output)} chunks to {self.config.output_file}")
        return output

    def chunk_pages(
        self,
        pages: list[dict],
        interlinks: dict[str, list[str]] | None = None,
        page_types: dict[str, dict] | None = None,
    ) -> list[dict]:
        """Chunk a list of pages (satisfies ChunkerProtocol).

        This is the protocol-compatible entry point. Unlike ``chunk_all``
        it does NOT read files from disk — the caller supplies everything.
        """
        interlinks = interlinks or {}
        page_types = page_types or {}

        all_chunks: list[Chunk] = []
        for page in pages:
            title = page.get("title", "")
            cls = page_types.get(title, {})
            page_type = cls.get("page_type", "other")
            categories = cls.get("semantic_categories", page.get("categories", []))
            related = interlinks.get(title, [])
            all_chunks.extend(self.chunk_page(page, page_type, categories, related))

        return [asdict(c) for c in all_chunks]


if __name__ == "__main__":
    chunker = Chunker()
    chunker.chunk_all()
