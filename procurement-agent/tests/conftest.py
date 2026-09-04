"""
tests/conftest.py

Shared pytest fixtures.

Mock strategy
-------------
- ChromaDB      : real chromadb.EphemeralClient (in-memory, no files written)
                  wrapped in a ChromaManager.  This exercises the real Chroma
                  code paths without any disk I/O.
- Groq / OpenAI : patched at the openai.OpenAI call site
- HuggingFace   : embed_texts / embed_query patched to return deterministic
                  zero-vectors so tests never download the model
- FastAPI app   : TestClient wrapping app.main:app; the chroma_client module's
                  get_chroma_manager singleton is replaced before the client
                  starts so every in-request call hits the in-memory store.
"""

from __future__ import annotations

import io
from typing import Generator
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import get_settings
from app.vector_store.chroma_client import ChromaManager

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


# ── In-memory ChromaDB manager ────────────────────────────────────────────────

def make_ephemeral_chroma_manager() -> ChromaManager:
    """
    Return a ChromaManager backed by a real chromadb.EphemeralClient.
    EphemeralClient stores everything in memory — no files, no server.
    """
    ephemeral_client = chromadb.EphemeralClient(
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    manager = ChromaManager(client=ephemeral_client)
    manager.ensure_collection()
    return manager


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_chroma() -> ChromaManager:
    """Session-scoped in-memory ChromaDB manager shared across all tests."""
    return make_ephemeral_chroma_manager()


@pytest.fixture(scope="session")
def client(mock_chroma) -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient with:
      - ChromaDB replaced by in-memory EphemeralClient via ChromaManager
      - embed_texts / embed_query patched to return zero-vectors
      - openai.OpenAI patched to return a canned LLM response
    """
    from app.vector_store import chroma_client as cc_module

    # Replace the cached singleton so every API call hits in-memory Chroma
    cc_module.get_chroma_manager.cache_clear()
    cc_module.get_chroma_manager = lambda: mock_chroma  # type: ignore[assignment]

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


# ── Sample document byte fixtures ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """Minimal valid single-page PDF bytes (no external file needed)."""
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
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client
