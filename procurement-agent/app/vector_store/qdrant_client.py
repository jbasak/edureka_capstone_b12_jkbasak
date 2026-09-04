"""
app/vector_store/qdrant_client.py

Qdrant vector database interface.

Responsibilities
----------------
- ensure_collection()  : create the collection if it does not exist yet.
- upsert_chunks()      : idempotent bulk insert of TextChunk objects.
- search()             : cosine-similarity search with optional vendor filter.
- collection_info()    : metadata for the /health endpoint.
- delete_by_source()   : remove all vectors for a given source_file.

Point-ID strategy
-----------------
Each point ID is derived from sha256(source_file + str(chunk_index)), truncated
to 63 bits and cast to a Python int.  This makes re-uploads idempotent: the same
file produces the same IDs and Qdrant's upsert overwrites existing points.

Payload schema (stored alongside each vector)
---------------------------------------------
  source_file   : str
  vendor_name   : str | None
  doc_category  : str | None
  page_number   : int | None
  row_index     : int | None
  sheet_name    : str | None
  chunk_index   : int
  total_chunks  : int
  token_count   : int
  text          : str   ← stored for validation + citation display
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)


# ── Helper: stable point ID from filename + chunk index ──────────────────────

def _point_id(source_file: str, chunk_index: int) -> int:
    """Return a 63-bit int derived from sha256(source_file + chunk_index)."""
    digest = hashlib.sha256(f"{source_file}::{chunk_index}".encode()).digest()
    # Use first 8 bytes; mask to 63 bits to stay within signed int64 range
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


# ── Manager class ─────────────────────────────────────────────────────────────

class QdrantManager:
    """
    Thin wrapper around qdrant_client.QdrantClient that encapsulates all
    collection management and vector operations for this application.
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._client = QdrantClient(
            host=self._settings.qdrant_host,
            port=self._settings.qdrant_port,
            timeout=30,
        )
        self._collection = self._settings.qdrant_collection
        self._dim = self._settings.vector_dimension
        logger.info(
            "QdrantManager initialised: %s:%d / collection='%s'",
            self._settings.qdrant_host,
            self._settings.qdrant_port,
            self._collection,
        )

    # ── Collection management ─────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection in existing:
            logger.info("Collection '%s' already exists.", self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=self._dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' (dim=%d, distance=COSINE).",
            self._collection,
            self._dim,
        )

    def collection_info(self) -> Optional[Any]:
        """Return collection info object, or None if unavailable."""
        try:
            return self._client.get_collection(self._collection)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch collection info: %s", exc)
            return None

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        vectors: list[list[float]],
    ) -> int:
        """
        Insert (or overwrite) a batch of TextChunk objects into Qdrant.

        Parameters
        ----------
        chunks  : list of TextChunk metadata objects
        vectors : parallel list of embedding vectors

        Returns
        -------
        Number of points upserted.

        Raises
        ------
        ValueError : if chunks and vectors lengths differ
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have the same length."
            )

        if not chunks:
            return 0

        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = _point_id(chunk.source_file, chunk.chunk_index)
            payload = {
                "source_file": chunk.source_file,
                "vendor_name": chunk.vendor_name,
                "doc_category": chunk.doc_category,
                "page_number": chunk.page_number,
                "row_index": chunk.row_index,
                "sheet_name": chunk.sheet_name,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "token_count": chunk.token_count,
                "text": chunk.text,
            }
            points.append(
                qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        # Qdrant upsert is idempotent — existing IDs are overwritten
        self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,  # synchronous — ensures 201 means stored
        )
        logger.info(
            "Upserted %d points into '%s' for source='%s'.",
            len(points),
            self._collection,
            chunks[0].source_file if chunks else "?",
        )
        return len(points)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 6,
        vendor_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Cosine-similarity search with optional vendor filter.

        Parameters
        ----------
        query_vector  : embedded query (384-dim float list)
        top_k         : number of results to return
        vendor_filter : if provided, restrict to these vendor_name values

        Returns
        -------
        List of dicts with keys: score, source_file, vendor_name, page_number,
        row_index, chunk_index, text (and all other payload keys).
        """
        qdrant_filter = None
        if vendor_filter:
            qdrant_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="vendor_name",
                        match=qmodels.MatchAny(any=vendor_filter),
                    )
                ]
            )

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        hits = []
        for hit in results:
            entry = {"score": hit.score}
            entry.update(hit.payload or {})
            hits.append(entry)

        logger.debug(
            "Search returned %d hits (top_k=%d, vendor_filter=%s).",
            len(hits),
            top_k,
            vendor_filter,
        )
        return hits

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_by_source(self, source_file: str) -> int:
        """
        Remove all vectors belonging to a specific source file.

        Returns the number of deleted points (Qdrant does not expose this
        directly; we return -1 to indicate the operation was attempted).
        """
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source_file",
                            match=qmodels.MatchValue(value=source_file),
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info("Deleted vectors for source_file='%s'.", source_file)
        return -1  # Qdrant does not return count on delete


# ── Module-level singleton ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_qdrant_manager() -> QdrantManager:
    """Return a cached QdrantManager singleton."""
    return QdrantManager()
