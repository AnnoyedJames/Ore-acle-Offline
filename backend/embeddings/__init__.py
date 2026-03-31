"""Embedding module — factory for local and API-based embedders."""

from backend.config.settings import EMBEDDING_MODELS, settings


def get_embedder(model_id: str | None = None):
    """Return the appropriate embedder for *model_id*.

    Parameters
    ----------
    model_id : str | None
        Key from :data:`EMBEDDING_MODELS`.  Falls back to
        ``settings.embedding_model`` when *None*.

    Returns
    -------
    EmbeddingGenerator | ApiEmbeddingGenerator
        An embedder satisfying ``embed_passages`` / ``embed_query`` / ``dimension``.
    """
    model_id = model_id or settings.embedding_model
    info = EMBEDDING_MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"Unknown embedding model '{model_id}'. "
            f"Available: {list(EMBEDDING_MODELS)}"
        )

    if info.backend == "api":
        from backend.embeddings.api_generator import ApiEmbeddingGenerator

        return ApiEmbeddingGenerator(model_id=model_id)

    # Local sentence-transformers backend
    from backend.embeddings.generator import EmbeddingConfig, EmbeddingGenerator

    config = EmbeddingConfig(
        model_name=model_id,
        truncate_dim=info.dimension,
        task_prefix=info.task_prefix,
        query_prefix=info.query_prefix,
    )
    return EmbeddingGenerator(config=config)
