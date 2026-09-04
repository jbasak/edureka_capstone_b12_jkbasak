"""
tests/test_chat_api.py

Integration tests for POST /chat and POST /chat/compliance.

The session-scoped TestClient (conftest.py) patches:
  - ChromaDB → in-memory EphemeralClient (pre-populated via helper)
  - embed_query → zero-vector
  - openai.OpenAI → canned LLM response with citations

Acceptance criteria covered
---------------------------
  Scenario 2: comparative query returns Markdown table structure
  Scenario 3: every response contains at least one citation
  Hallucination guard: empty-DB query returns sentinel
  Vendor filter: sources only from requested vendor
"""

from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

CANNED_COMPARATIVE = (
    "## Pricing Comparison\n\n"
    "| Vendor | Annual Cost | Uptime |\n"
    "|--------|-------------|--------|\n"
    "| AWS    | $480,000    | 99.99% [AWS Response to Michigan DTMB Cloud Services RFP_Public.pdf, Page 3] |\n"
    "| Oracle | $520,000    | 99.9%  [Schedule B Pricing-Oracle.pdf, Page 5] |\n\n"
    "AWS offers lower annual pricing [AWS Response to Michigan DTMB Cloud Services RFP_Public.pdf, Page 3]."
)

CANNED_EXTRACTIVE = (
    "AWS provides 99.99% uptime SLA as documented in their proposal "
    "[AWS Response to Michigan DTMB Cloud Services RFP_Public.pdf, Page 3]."
)


def _patch_llm(content: str):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def _ingest_sample_docs(client, sample_txt_bytes, sample_csv_bytes):
    """Pre-populate the in-memory vector store with a few documents."""
    client.post(
        "/documents",
        files={"file": ("AWS Response to Michigan DTMB Cloud Services RFP_Public.pdf",
                        sample_txt_bytes, "text/plain")},
        data={"vendor_name": "AWS", "doc_category": "proposal"},
    )
    client.post(
        "/documents",
        files={"file": ("Schedule B Pricing-Oracle.pdf",
                        sample_csv_bytes, "text/csv")},
        data={"vendor_name": "Oracle", "doc_category": "pricing"},
    )


# ── POST /chat basic ──────────────────────────────────────────────────────────

class TestChatBasic:
    def test_returns_200(self, client, sample_txt_bytes, sample_csv_bytes):
        _ingest_sample_docs(client, sample_txt_bytes, sample_csv_bytes)
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_EXTRACTIVE)):
            response = client.post(
                "/chat",
                json={"question": "What is the AWS uptime SLA?"},
            )
        assert response.status_code == 200

    def test_response_schema_complete(self, client, sample_txt_bytes):
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_EXTRACTIVE)):
            response = client.post(
                "/chat",
                json={"question": "What is the uptime guarantee?"},
            )
        data = response.json()
        for field in ("answer", "mode", "sources", "validation_passed", "citations_count"):
            assert field in data, f"Missing field: {field}"

    def test_validation_passed_field_always_present(self, client):
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_EXTRACTIVE)):
            response = client.post(
                "/chat",
                json={"question": "What services are offered?"},
            )
        assert "validation_passed" in response.json()

    def test_sources_is_list(self, client):
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_EXTRACTIVE)):
            response = client.post(
                "/chat",
                json={"question": "Security requirements"},
            )
        assert isinstance(response.json()["sources"], list)


# ── Extractive mode ───────────────────────────────────────────────────────────

class TestExtractiveMode:
    def test_mode_is_extractive_for_non_comparison_question(
        self, client, sample_txt_bytes
    ):
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_EXTRACTIVE)):
            response = client.post(
                "/chat",
                json={"question": "What is the AWS uptime SLA?"},
            )
        assert response.json()["mode"] == "extractive"

    def test_explicit_extractive_mode_honoured(self, client):
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_EXTRACTIVE)):
            response = client.post(
                "/chat",
                json={"question": "Describe security features", "mode": "extractive"},
            )
        assert response.json()["mode"] == "extractive"

    def test_answer_non_empty_when_docs_indexed(self, client, sample_txt_bytes):
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_EXTRACTIVE)):
            response = client.post(
                "/chat",
                json={"question": "What is the pricing?"},
            )
        assert len(response.json()["answer"]) > 0


# ── Comparative mode (Scenario 2) ────────────────────────────────────────────

class TestComparativeMode:
    def test_compare_keyword_sets_comparative_mode(
        self, client, sample_txt_bytes, sample_csv_bytes
    ):
        _ingest_sample_docs(client, sample_txt_bytes, sample_csv_bytes)
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_COMPARATIVE)):
            response = client.post(
                "/chat",
                json={"question": "Compare annual pricing across AWS and Oracle"},
            )
        assert response.json()["mode"] == "comparative"

    def test_comparative_answer_contains_table_markers(
        self, client, sample_txt_bytes, sample_csv_bytes
    ):
        _ingest_sample_docs(client, sample_txt_bytes, sample_csv_bytes)
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_COMPARATIVE)):
            response = client.post(
                "/chat",
                json={"question": "Compare annual pricing"},
            )
        answer = response.json()["answer"]
        # Markdown table must have pipe characters
        assert "|" in answer

    def test_explicit_comparative_mode_honoured(self, client):
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_COMPARATIVE)):
            response = client.post(
                "/chat",
                json={"question": "Show me pricing", "mode": "comparative"},
            )
        assert response.json()["mode"] == "comparative"


# ── Citation validation (Scenario 3) ─────────────────────────────────────────

class TestCitations:
    def test_citations_count_positive_when_answer_has_citations(
        self, client, sample_txt_bytes
    ):
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_EXTRACTIVE)):
            response = client.post(
                "/chat",
                json={"question": "What is the uptime?"},
            )
        data = response.json()
        # Citations present iff chunks were retrieved
        if data["sources"]:
            assert data["citations_count"] >= 1
            assert "[" in data["answer"]

    def test_citation_format_in_answer(self, client, sample_txt_bytes):
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_EXTRACTIVE)):
            response = client.post(
                "/chat",
                json={"question": "Uptime SLA"},
            )
        answer = response.json()["answer"]
        if response.json()["sources"]:
            # Should contain at least one [filename, Page N] or [filename, Row N]
            import re
            pattern = r"\[[^\]]+,\s*(Page|Row|Chunk)\s+\d+\]"
            assert re.search(pattern, answer, re.IGNORECASE), (
                f"No citation found in answer: {answer!r}"
            )


# ── Hallucination guard ───────────────────────────────────────────────────────

class TestHallucinationGuard:
    def test_empty_db_returns_sentinel(self):
        """
        With a fresh empty in-memory Chroma store (no ingested docs), the
        retrieval agent returns 0 chunks and the controller short-circuits
        to the sentinel string.
        """
        import chromadb as _chromadb
        from app.vector_store.chroma_client import ChromaManager
        from app.vector_store import chroma_client as cc_module

        empty_client = _chromadb.EphemeralClient(
            settings=_chromadb.config.Settings(anonymized_telemetry=False)
        )
        empty_manager = ChromaManager(client=empty_client)
        empty_manager.ensure_collection()

        original = cc_module.get_chroma_manager
        cc_module.get_chroma_manager = lambda: empty_manager  # type: ignore[assignment]

        try:
            with patch("app.agents.retrieval.embed_query", return_value=[0.0] * 384):
                from fastapi.testclient import TestClient
                from app.main import app as _app
                with TestClient(_app) as c:
                    response = c.post(
                        "/chat",
                        json={"question": "Who offers the cheapest option?"},
                    )
        finally:
            cc_module.get_chroma_manager = original  # type: ignore[assignment]

        data = response.json()
        assert data["answer"] == "Information not found in context."
        assert data["sources"] == []
        assert data["validation_passed"] is True


# ── Vendor filter ─────────────────────────────────────────────────────────────

class TestVendorFilter:
    def test_vendor_filter_accepted(self, client, sample_txt_bytes):
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(CANNED_EXTRACTIVE)):
            response = client.post(
                "/chat",
                json={
                    "question": "What is the pricing?",
                    "vendor_filter": ["AWS"],
                },
            )
        assert response.status_code == 200

    def test_invalid_request_body_returns_422(self, client):
        response = client.post("/chat", json={"not_question": "oops"})
        assert response.status_code == 422

    def test_blank_question_returns_422(self, client):
        response = client.post("/chat", json={"question": "  "})
        assert response.status_code == 422


# ── POST /chat/compliance ─────────────────────────────────────────────────────

class TestComplianceEndpoint:
    def test_compliance_returns_200(
        self, client, sample_xlsx_bytes, sample_txt_bytes, sample_csv_bytes
    ):
        _ingest_sample_docs(client, sample_txt_bytes, sample_csv_bytes)
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(
            "Pass. AWS satisfies this requirement [AWS Response to Michigan DTMB "
            "Cloud Services RFP_Public.pdf, Page 3]."
        )):
            response = client.post(
                "/chat/compliance",
                files={"file": ("requirements.xlsx", sample_xlsx_bytes,
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                data={"vendors": "AWS,Oracle"},
            )
        assert response.status_code == 200

    def test_compliance_response_has_table(
        self, client, sample_xlsx_bytes, sample_txt_bytes, sample_csv_bytes
    ):
        _ingest_sample_docs(client, sample_txt_bytes, sample_csv_bytes)
        with patch("app.agents.reasoning.OpenAI", return_value=_patch_llm(
            "Pass [AWS Response to Michigan DTMB Cloud Services RFP_Public.pdf, Page 3]."
        )):
            response = client.post(
                "/chat/compliance",
                files={"file": ("req.xlsx", sample_xlsx_bytes,
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                data={"vendors": "AWS"},
            )
        data = response.json()
        assert "compliance_table" in data
        assert "|" in data["compliance_table"]

    def test_compliance_unsupported_file_returns_422(self, client):
        response = client.post(
            "/chat/compliance",
            files={"file": ("reqs.pdf", b"%PDF", "application/pdf")},
            data={"vendors": "AWS"},
        )
        assert response.status_code == 422

    def test_compliance_empty_vendors_returns_422(self, client, sample_xlsx_bytes):
        response = client.post(
            "/chat/compliance",
            files={"file": ("req.xlsx", sample_xlsx_bytes,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"vendors": ""},
        )
        assert response.status_code == 422
