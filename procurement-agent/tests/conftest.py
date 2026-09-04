"""
tests/conftest.py

Shared pytest fixtures.

Mock strategy
-------------
- Qdrant        : replaced with an in-memory dict store (no Docker needed)
- Groq / OpenAI : patched at the openai.OpenAI call site
- HuggingFace   : embed_texts / embed_query patched to return deterministic
                  zero-vectors so tests never download the model
- FastAPI app   : TestClient wrapping app.main:app with dependency overrides
                  that inject the mock Qdrant manager
"""

from __future__ import annotations

import io
import json
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import get_settings

settings = get_settings()

# ── Deterministic vector helpers ──────────────────────────────────────────────

VECTOR_DIM = 384


def zero_vector() -> list[float]:
    return [0.0] * VECTOR_DIM


def unit_vector(seed: int = 1) -> list[float]:
    """A non-zero vector useful for unique-but-stable embeddings."""
    v = [0.0] * VECTOR_DIM
    v[seed % VECTOR_DIM] = 1.0
    return v


# ── In-memory Qdrant mock ─────────────────────────────────────────────────────

class InMemoryQdrantManager:
    """
    Drop-in replacement for QdrantManager that stores vectors in a plain dict.
    Supports: ensure_collection, upsert_chunks, search, delete_by_source,
              collection_info, and the _client.scroll interface used by GET /documents.
    """

    def __init__(self):
        self._store: dict[int, dict] = {}  # point_id → {vector, payload}
        self._client = _ScrollAdapter(self._store)

    def ensure_collection(self):
        pass

    def collection_info(self):
        info = MagicMock()
        info.vectors_count = len(self._store)
        return info

    def upsert_chunks(self, chunks, vectors):
        import hashlib

        for chunk, vec in zip(chunks, vectors):
            digest = hashlib.sha256(
                f"{chunk.source_file}::{chunk.chunk_index}".encode()
            ).digest()
            pid = int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
            self._store[pid] = {
                "vector": vec,
                "payload": {
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
                },
            }
        return len(chunks)

    def search(self, query_vector, top_k=6, vendor_filter=None):
        results = []
        for pid, entry in self._store.items():
            payload = entry["payload"]
            if vendor_filter and payload.get("vendor_name") not in vendor_filter:
                continue
            results.append({"score": 0.85, **payload})
        return results[:top_k]

    def delete_by_source(self, source_file: str):
        to_delete = [
            pid for pid, entry in self._store.items()
            if entry["payload"].get("source_file") == source_file
        ]
        for pid in to_delete:
            del self._store[pid]
        return len(to_delete)


class _ScrollAdapter:
    """Minimal mock of qdrant_client.QdrantClient.scroll() for GET /documents."""

    def __init__(self, store: dict):
        self._store = store

    def scroll(self, collection_name, limit=256, offset=None,
               with_payload=None, with_vectors=False):
        points = []
        for pid, entry in self._store.items():
            pt = MagicMock()
            pt.payload = entry["payload"]
            points.append(pt)
        # Return all at once (no pagination needed for tests)
        return points, None


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_qdrant() -> InMemoryQdrantManager:
    """Session-scoped in-memory Qdrant manager shared across all tests."""
    return InMemoryQdrantManager()


@pytest.fixture(scope="session")
def client(mock_qdrant) -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient with:
      - Qdrant replaced by InMemoryQdrantManager
      - embed_texts / embed_query patched to return zero-vectors
      - openai.OpenAI patched to return a canned LLM response
    """
    from app.vector_store import qdrant_client as qc_module

    # Override the cached singleton so the API uses our in-memory store
    qc_module.get_qdrant_manager.cache_clear()
    qc_module.get_qdrant_manager = lambda: mock_qdrant  # type: ignore[assignment]

    with (
        patch("app.ingestion.embedder.embed_texts",
              side_effect=lambda texts: [zero_vector() for _ in texts]),
        patch("app.ingestion.embedder.embed_query",
              return_value=zero_vector()),
        patch("app.agents.retrieval.embed_query",
              return_value=zero_vector()),
        patch("openai.OpenAI", return_value=_mock_openai_client()),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """Minimal valid single-page PDF bytes (no external file needed)."""
    # Manually constructed minimal PDF
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>\nstream\nBT /F1 12 Tf 100 700 Td"
        b"(Sample RFP text for testing.) Tj ET\nendstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n430\n%%EOF"
    )


@pytest.fixture(scope="session")
def sample_txt_bytes() -> bytes:
    return (
        b"VENDOR PROPOSAL\n\n"
        b"Section 1: Uptime Guarantee\n"
        b"We guarantee 99.9% uptime for all cloud services.\n\n"
        b"Section 2: Security\n"
        b"All data is encrypted at rest and in transit using AES-256.\n\n"
        b"Section 3: Pricing\n"
        b"Annual cost for standard tier: $500,000.\n"
    )


@pytest.fixture(scope="session")
def sample_csv_bytes() -> bytes:
    return (
        b"vendor,service,annual_cost,uptime_sla\n"
        b"AWS,Compute,480000,99.99%\n"
        b"Google,Compute,460000,99.95%\n"
        b"Oracle,Database,520000,99.9%\n"
    )


@pytest.fixture(scope="session")
def sample_xlsx_bytes() -> bytes:
    """Create a minimal xlsx in-memory using openpyxl."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Requirements"
    ws.append(["ID", "Requirement", "Category"])
    ws.append(["REQ-001", "Vendor must provide 99.9% uptime SLA", "Availability"])
    ws.append(["REQ-002", "All data must be encrypted at rest", "Security"])
    ws.append(["REQ-003", "Vendor must support FedRAMP authorization", "Compliance"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ── Mock OpenAI / Groq client ─────────────────────────────────────────────────

def _mock_openai_client():
    """Return a MagicMock that looks like openai.OpenAI with a canned completion."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = (
        "Based on the provided context, AWS offers 99.99% uptime "
        "[AWS Response to Michigan DTMB Cloud Services RFP_Public.pdf, Page 3]. "
        "Oracle provides database services at $520,000 annually "
        "[Schedule B Pricing-Oracle.pdf, Page 5]."
    )
    client.chat.completions.create.return_value = MagicMock(
        choices=[choice]
    )
    return client
