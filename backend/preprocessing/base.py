"""
Preprocessing protocol definitions for swappable pipeline components.

These Protocols define the input/output contracts that chunkers (and
potentially other pluggable steps) must satisfy.  The concrete
implementations (SectionAwareChunker, future LangChain adapter, etc.)
simply need to expose the right method signatures — no base-class
inheritance required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


# ------------------------------------------------------------------
# Shared data structures
# ------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """Minimal output contract for a single retrieval chunk.

    Concrete chunkers may carry additional fields on their own
    dataclass, but every chunk piped into the embedding / indexing
    stages must at least satisfy this shape.
    """

    chunk_id: str
    page_title: str
    page_url: str
    section_heading: str
    text: str
    token_count: int
    chunk_type: str  # "section" | "infobox" | "table"
    images: list[dict] = field(default_factory=list)

    # Optional enrichment fields
    section_level: int = 2
    section_type: str = "content"
    page_type: str = "other"
    categories: list[str] = field(default_factory=list)
    related_pages: list[str] = field(default_factory=list)
    infobox: Optional[dict] = None


# ------------------------------------------------------------------
# Chunker protocol
# ------------------------------------------------------------------

@runtime_checkable
class ChunkerProtocol(Protocol):
    """Contract for any chunking implementation.

    Implementations must expose ``chunk_pages`` which accepts a list
    of processed page dicts (as produced by the text cleaner) together
    with optional enrichment data, and returns a flat list of chunk
    dicts ready for serialisation to ``chunks.json``.
    """

    def chunk_pages(
        self,
        pages: list[dict[str, Any]],
        interlinks: dict[str, list[str]] | None = None,
        page_types: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk a list of processed pages.

        Parameters
        ----------
        pages:
            List of page dicts from metadata.json (``pages`` key).
        interlinks:
            Optional page→[linked_pages] graph from interlinks.json.
        page_types:
            Optional page_title→classification from classified_pages.json.

        Returns
        -------
        list[dict]
            Flat list of chunk dicts.  Each dict should at minimum
            contain the fields of :class:`ChunkRecord`.
        """
        ...
