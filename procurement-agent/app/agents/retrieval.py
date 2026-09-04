"""
app/agents/retrieval.py

Retrieval Agent
---------------
Embeds the user question and pulls the top-K most relevant chunks from Qdrant.
Returns a ranked list of RetrievedChunk objects with scores and full metadata.

The agent is intentionally stateless: every call is independent so it can be
used from both the Agent Controller and the Requirement Extraction Agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.ingestion.embedder import embed_query
from app.vector_store.qdrant_client import get_qdrant_manager

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    """A single search hit returned by Qdrant with provenance metadata."""

    text: str
    source_file: str
    vendor_name: Optional[str]
    doc_category: Optional[str]
    page_number: Optional[int]
    row_index: Optional[int]
    sheet_name: Optional[str]
    chunk_index: int
    score: float

    def citation_label(self) -> str:
        """
        Format a bracketed citation string, e.g.:
          [AWS_Proposal.pdf, Page 3]
          [Recent Purchase History.csv, Row 12]
        """
        if self.page_number is not None:
            location = f"Page {self.page_number}"
        elif self.row_index is not None:
            location = f"Row {self.row_index}"
        else:
            location = f"Chunk {self.chunk_index}"
        return f"[{self.source_file}, {location}]"

    def short_excerpt(self, max_chars: int = 200) -> str:
        """Return the first *max_chars* characters of text for previews."""
        t = self.text.strip()
        return t[:max_chars] + ("…" if len(t) > max_chars else "")


class RetrievalAgent:
    """
    Wraps Qdrant search: embed → search → deserialise hits → rank.
    """

    def __init__(self, qdrant_manager=None, top_k: Optional[int] = None):
        self._qdrant = qdrant_manager or get_qdrant_manager()
        self._top_k = top_k or settings.top_k

    def retrieve(
        self,
        question: str,
        vendor_filter: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Embed *question* and return the top-K matching chunks from Qdrant.

        Parameters
        ----------
        question      : natural-language query string
        vendor_filter : optional list of vendor names to restrict results
        top_k         : override the default top-K value for this call

        Returns
        -------
        List of RetrievedChunk sorted by descending similarity score.
        Empty list when no documents are indexed.
        """
        k = top_k or self._top_k
        logger.info(
            "RetrievalAgent: embedding query (top_k=%d, vendor_filter=%s)",
            k,
            vendor_filter,
        )

        query_vector = embed_query(question)
        hits = self._qdrant.search(
            query_vector=query_vector,
            top_k=k,
            vendor_filter=vendor_filter,
        )

        chunks = []
        for h in hits:
            chunks.append(
                RetrievedChunk(
                    text=h.get("text", ""),
                    source_file=h.get("source_file", "unknown"),
                    vendor_name=h.get("vendor_name"),
                    doc_category=h.get("doc_category"),
                    page_number=h.get("page_number"),
                    row_index=h.get("row_index"),
                    sheet_name=h.get("sheet_name"),
                    chunk_index=h.get("chunk_index", 0),
                    score=round(float(h.get("score", 0.0)), 4),
                )
            )

        logger.info("RetrievalAgent: returned %d chunks.", len(chunks))
        return chunks
