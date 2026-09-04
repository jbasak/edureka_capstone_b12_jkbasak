"""
app/vector_store/chroma_client.py

ChromaDB vector database interface.

Why ChromaDB instead of Qdrant
-------------------------------
ChromaDB is a pure-Python, embedded vector database.  It requires no separate
server process, no Docker, and no network configuration — it persists directly
to a local directory specified by CHROMA_PERSIST_DIR in .env.

The public API of this module is intentionally identical to the old
qdrant_client.py so no agent or router code changes are needed.

Public interface
----------------
  ChromaManager.ensure_collection()      create collection if missing
  ChromaManager.upsert_chunks()          idempotent bulk insert
  ChromaManager.search()                 cosine similarity search with optional vendor filter
  ChromaManager.delete_by_source()       remove all chunks for a source file
  ChromaManager.collection_info()        metadata dict for /health endpoint
  ChromaManager.list_sources()           all (source_file, count) pairs

  get_chroma_manager()                   module-level cached singleton

Point-ID strategy
-----------------
Identical to the former Qdrant implementation: IDs are derived from
sha256(source_file + "::" + str(chunk_index)), hex-encoded so they are
valid Chroma string IDs.  Re-uploading the same file produces the same IDs
and Chroma's upsert overwrites without duplicating.

Metadata stored per document
-----------------------------
  source_file, vendor_name, doc_category, page_number, row_index,
  sheet_name, chunk_index, total_chunks, token_count, text
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)


# ── Stable string ID ──────────────────────────────────────────────────────────

def _point_id(source_file: str, chunk_index: int) -> str:
    """Return a hex-string ID derived from sha256(source_file::chunk_index)."""
    return hashlib.sha256(
        f"{source_file}::{chunk_index}".encode()
    ).hexdigest()


# ── ChromaDB metadata helpers ─────────────────────────────────────────────────

def _to_metadata(chunk: TextChunk) -> dict:
    """
    Build a flat metadata dict for ChromaDB.
    ChromaDB only accepts str / int / float / bool values — None is not allowed,
    so we substitute empty string for optional string fields and -1 for optional
    int fields.
    """
    return {
        "source_file": chunk.source_file,
        "vendor_name": chunk.vendor_name or "",
        "doc_category": chunk.doc_category or "",
        "page_number": chunk.page_number if chunk.page_number is not None else -1,
        "row_index": chunk.row_index if chunk.row_index is not None else -1,
        "sheet_name": chunk.sheet_name or "",
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "token_count": chunk.token_count,
        "text": chunk.text,
    }


def _from_metadata(meta: dict, score: float) -> dict:
    """
    Convert a raw ChromaDB metadata dict back to the shape the rest of the app
    expects (None for sentinel values, score attached).
    """
    return {
        "score": round(score, 4),
        "source_file": meta.get("source_file", "unknown"),
        "vendor_name": meta.get("vendor_name") or None,
        "doc_category": meta.get("doc_category") or None,
        "page_number": meta.get("page_number") if meta.get("page_number", -1) != -1 else None,
        "row_index": meta.get("row_index") if meta.get("row_index", -1) != -1 else None,
        "sheet_name": meta.get("sheet_name") or None,
        "chunk_index": meta.get("chunk_index", 0),
        "total_chunks": meta.get("total_chunks", 1),
        "token_count": meta.get("token_count", 0),
        "text": meta.get("text", ""),
    }


# ── Manager class ─────────────────────────────────────────────────────────────

class ChromaManager:
    """
    Thin wrapper around chromadb.PersistentClient (or EphemeralClient for tests)
    that exposes the same interface as the former QdrantManager.
    """

    def __init__(self, settings=None, client: Optional[chromadb.ClientAPI] = None):
        self._settings = settings or get_settings()

        if client is not None:
            # Injected client (used by tests for in-memory ephemeral store)
            self._client = client
        else:
            persist_dir = self._settings.chroma_persist_dir
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(
                "ChromaManager: persistent store at '%s'", persist_dir
            )

        self._collection_name = self._settings.chroma_collection
        self._collection: Optional[chromadb.Collection] = None

    # ── Collection management ─────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """
        Get or create the ChromaDB collection.
        Uses cosine distance to match the former Qdrant configuration.
        """
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        count = self._collection.count()
        logger.info(
            "ChromaDB collection '%s' ready (%d documents).",
            self._collection_name,
            count,
        )

    def _get_collection(self) -> chromadb.Collection:
        """Return collection, initialising lazily if ensure_collection was not called."""
        if self._collection is None:
            self.ensure_collection()
        return self._collection  # type: ignore[return-value]

    def collection_info(self) -> dict:
        """Return a dict with collection name and document count (used by /health)."""
        try:
            col = self._get_collection()
            return {
                "name": self._collection_name,
                "vectors_count": col.count(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("collection_info failed: %s", exc)
            return {"name": self._collection_name, "vectors_count": None}

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        vectors: list[list[float]],
    ) -> int:
        """
        Idempotent bulk insert.  Existing IDs are overwritten.

        Parameters
        ----------
        chunks  : TextChunk objects produced by the chunker
        vectors : parallel embedding vectors

        Returns
        -------
        Number of documents upserted.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) lengths differ."
            )
        if not chunks:
            return 0

        col = self._get_collection()

        ids = [_point_id(c.source_file, c.chunk_index) for c in chunks]
        metadatas = [_to_metadata(c) for c in chunks]
        documents = [c.text for c in chunks]

        # ChromaDB upsert: add if new, update if ID already exists
        col.upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=documents,
        )

        logger.info(
            "ChromaDB: upserted %d chunks for source='%s'.",
            len(chunks),
            chunks[0].source_file,
        )
        return len(chunks)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 6,
        vendor_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Cosine-similarity search with optional vendor_name filter.

        Parameters
        ----------
        query_vector  : 384-dim query embedding
        top_k         : maximum results to return
        vendor_filter : list of vendor names to restrict results

        Returns
        -------
        List of dicts with keys: score, source_file, vendor_name, text, etc.
        Empty list if collection has no documents.
        """
        col = self._get_collection()

        if col.count() == 0:
            return []

        # Build ChromaDB where-clause for vendor filter
        where: Optional[dict] = None
        if vendor_filter:
            if len(vendor_filter) == 1:
                where = {"vendor_name": {"$eq": vendor_filter[0]}}
            else:
                where = {"vendor_name": {"$in": vendor_filter}}

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, col.count()),
            "include": ["metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = col.query(**query_kwargs)

        hits = []
        metadatas_list = results.get("metadatas") or [[]]
        distances_list = results.get("distances") or [[]]

        for meta, distance in zip(
            metadatas_list[0],
            distances_list[0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity score in [0, 1]:  score = 1 - distance/2
            score = max(0.0, 1.0 - distance / 2.0)
            hits.append(_from_metadata(meta, score))

        logger.debug(
            "ChromaDB search: %d hits (top_k=%d, vendor_filter=%s)",
            len(hits),
            top_k,
            vendor_filter,
        )
        return hits

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_by_source(self, source_file: str) -> int:
        """
        Remove all documents whose source_file metadata matches the given value.

        Returns the number of deleted documents.
        """
        col = self._get_collection()

        # Query IDs matching this source first (Chroma requires IDs for delete)
        results = col.get(
            where={"source_file": {"$eq": source_file}},
            include=[],  # IDs only
        )
        ids_to_delete = results.get("ids", [])

        if ids_to_delete:
            col.delete(ids=ids_to_delete)
            logger.info(
                "ChromaDB: deleted %d documents for source_file='%s'.",
                len(ids_to_delete),
                source_file,
            )
        return len(ids_to_delete)

    # ── List sources ──────────────────────────────────────────────────────────

    def list_sources(self) -> list[dict]:
        """
        Return all distinct source files with chunk counts.
        Used by GET /documents.
        """
        col = self._get_collection()

        if col.count() == 0:
            return []

        # Retrieve all metadata (no vectors needed)
        results = col.get(include=["metadatas"])
        source_counts: dict[str, int] = {}

        for meta in results.get("metadatas", []):
            src = (meta or {}).get("source_file", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        return [
            {"source_file": src, "chunks": count}
            for src, count in sorted(source_counts.items())
        ]


# ── Module-level singleton ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_chroma_manager() -> ChromaManager:
    """Return a cached ChromaManager singleton."""
    return ChromaManager()
