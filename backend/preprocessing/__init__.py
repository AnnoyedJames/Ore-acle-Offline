"""Preprocessing module — chunker factory."""

from backend.config.settings import settings


def get_chunker(strategy: str | None = None):
    """Return the chunker for the given strategy.

    Parameters
    ----------
    strategy : str | None
        ``"section_aware"`` (default) or ``"langchain"``.
        Falls back to ``settings.chunker_strategy``.

    Returns
    -------
    Chunker | LangChainChunker
    """
    strategy = strategy or settings.chunker_strategy

    if strategy == "langchain":
        from backend.preprocessing.langchain_chunker import LangChainChunker
        return LangChainChunker()

    # Default: section-aware
    from backend.preprocessing.chunker import Chunker
    return Chunker()
