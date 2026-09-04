"""
app/ingestion/embedder.py

HuggingFace sentence-transformers embedding wrapper.

Model : sentence-transformers/all-MiniLM-L6-v2
Output: 384-dimensional dense float vectors

Design notes
------------
- Model is loaded once on first call and cached as a module-level singleton
  to avoid repeated disk I/O on every request.
- Weights are stored in the HuggingFace cache directory, which is mounted as
  a Docker named volume so the model survives container restarts.
- Texts are encoded in batches of BATCH_SIZE (default 32) to keep GPU/CPU
  memory usage bounded when ingesting large documents.
- Returns plain Python lists of floats (not numpy arrays) so they serialise
  cleanly to JSON / ChromaDB metadata payloads.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from app.config import get_settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 32  # sentences per encoding batch


# ── Model singleton ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model():
    """
    Load the sentence-transformer model once and cache it.
    The lru_cache ensures a single load per process lifetime.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore

    model_name = get_settings().embedding_model
    logger.info("Loading embedding model '%s' …", model_name)
    model = SentenceTransformer(model_name)
    logger.info(
        "Embedding model loaded. Output dimension: %d",
        model.get_sentence_embedding_dimension(),
    )
    return model


# ── Public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> List[List[float]]:
    """
    Encode a list of text strings into embedding vectors.

    Parameters
    ----------
    texts : list of strings to embed (can be empty)

    Returns
    -------
    List of float vectors, one per input string.
    Each vector has length == settings.vector_dimension (384).

    Raises
    ------
    ValueError : if texts is not a list
    """
    if not isinstance(texts, list):
        raise ValueError(f"embed_texts expects a list, got {type(texts)}")

    if not texts:
        return []

    model = _load_model()
    vectors: List[List[float]] = []

    # Process in batches
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity ≡ dot product when normalised
        )
        vectors.extend(embedding.tolist() for embedding in embeddings)

    logger.debug("Embedded %d texts → %d vectors (dim=%d)", len(texts), len(vectors), len(vectors[0]))
    return vectors


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.  Convenience wrapper around embed_texts.

    Parameters
    ----------
    query : user question string

    Returns
    -------
    Single float vector of length 384.
    """
    if not query or not query.strip():
        raise ValueError("Query string must not be empty.")

    result = embed_texts([query.strip()])
    return result[0]
