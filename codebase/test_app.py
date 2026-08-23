"""
tests/test_app.py
==================
Automated test suite for the Procurement Multi-Agent RAG System (app.py).

Design notes:
- The real sentence-transformers model is NOT downloaded during these tests.
  We monkeypatch VectorStore.embed / VectorStore._lazy_load_model at the class
  level with a fast, deterministic hash-based embedding so tests are quick and
  work offline / in CI without network access.
- Since GROQ_API_KEY is unset by default in test environments, the
  ComparisonReasoningAgent naturally exercises its extractive fallback path.
  A couple of tests explicitly monkeypatch LLMClient.complete to verify the
  LLM-backed path and the ValidationAgent's citation checking.

Run with:
    pytest tests/ -v
"""

import io
import hashlib
import importlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import (
    app as fastapi_app,
    VectorStore,
    RequirementExtractionAgent,
    ValidationAgent,
    ComparisonReasoningAgent,
    LLMClient,
    DocumentProcessingError,
    chunk_text,
    Chunk,
    DocType,
    settings,
)

# ==============================================================================
# Fixtures
# ==============================================================================

def _fake_lazy_load_model(self):
    """Avoids downloading a real embedding model during tests."""
    if self._dim is None:
        self._dim = 32
    if app_module.faiss is not None and self._index is None:
        self._index = app_module.faiss.IndexFlatIP(self._dim)


def _fake_embed(self, texts):
    """Deterministic hash-based 'embedding' — identical text -> identical
    vector, different text -> (effectively) different vector. Good enough
    to exercise retrieval/storage logic without a real model."""
    vecs = []
    for t in texts:
        h = hashlib.md5(t.encode("utf-8")).digest()
        raw = (h * 4)[:32]
        arr = np.frombuffer(raw, dtype=np.uint8).astype("float32")
        arr = arr / (np.linalg.norm(arr) + 1e-8)
        vecs.append(arr)
    return np.vstack(vecs).astype("float32")


@pytest.fixture(autouse=True)
def patch_embeddings(monkeypatch):
    """Applied to every test: patches the VectorStore class (not just one
    instance) so it survives the /reset endpoint creating a fresh store."""
    monkeypatch.setattr(VectorStore, "_lazy_load_model", _fake_lazy_load_model)
    monkeypatch.setattr(VectorStore, "embed", _fake_embed)
    yield


@pytest.fixture(autouse=True)
def clean_state():
    """Ensures each test starts with an empty document/requirement store."""
    client = TestClient(fastapi_app)
    client.delete("/reset")
    yield
    client.delete("/reset")


@pytest.fixture
def client():
    return TestClient(fastapi_app)


def make_csv_bytes(rows):
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def make_excel_bytes(rows):
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


# ==============================================================================
# Unit tests — Chunking
# ==============================================================================

class TestChunking:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short vendor proposal summary."
        chunks = chunk_text(text, chunk_size=900, overlap=150)
        assert chunks == [text]

    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("   ", chunk_size=900, overlap=150) == []

    def test_long_text_splits_into_multiple_chunks(self):
        paragraph = "Vendor pricing details and terms. " * 40  # ~1400 chars
        text = "\n\n".join([paragraph] * 3)
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1
        # every chunk should respect (roughly) the size budget plus overlap
        assert all(len(c) <= 500 + 60 for c in chunks)

    def test_overlap_preserves_context_between_chunks(self):
        paragraph = "Clause A. " * 100
        text = "\n\n".join([paragraph, paragraph, paragraph])
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) >= 2
        # second chunk should contain trailing characters from the first
        tail_of_first = chunks[0][-20:].strip()
        assert any(tail_of_first[-10:] in c for c in chunks[1:])


# ==============================================================================
# Unit tests — Requirement Extraction Agent
# ==============================================================================

class TestRequirementExtractionAgent:
    def test_extracts_requirements_with_standard_columns(self):
        df = pd.DataFrame({
            "Requirement ID": ["REQ-001", "REQ-002"],
            "Description": ["Must support SSO", "Must provide 24/7 support"],
            "Mandatory (Y/N)": ["Y", "N"],
            "Category": ["Security", "Support"],
        })
        reqs = RequirementExtractionAgent().run(df, source_doc="requirements.xlsx")
        assert len(reqs) == 2
        assert reqs[0].req_id == "REQ-001"
        assert reqs[0].mandatory is True
        assert reqs[1].mandatory is False
        assert reqs[0].category == "Security"

    def test_handles_alternate_column_names(self):
        df = pd.DataFrame({
            "ID": [1, 2],
            "Requirement": ["Data must be encrypted at rest", "API uptime >= 99.9%"],
            "Required": ["Yes", "No"],
        })
        reqs = RequirementExtractionAgent().run(df, source_doc="reqs.xlsx")
        assert len(reqs) == 2
        assert reqs[0].mandatory is True
        assert reqs[1].mandatory is False

    def test_defaults_to_mandatory_when_no_mandatory_column(self):
        df = pd.DataFrame({"Description": ["Must integrate with SAP"]})
        reqs = RequirementExtractionAgent().run(df, source_doc="reqs.xlsx")
        assert len(reqs) == 1
        assert reqs[0].mandatory is True
        assert reqs[0].req_id == "REQ-001"  # auto-generated

    def test_skips_blank_description_rows(self):
        df = pd.DataFrame({"Description": ["Valid requirement", "", None]})
        reqs = RequirementExtractionAgent().run(df, source_doc="reqs.xlsx")
        assert len(reqs) == 1

    def test_raises_when_no_description_column_found(self):
        df = pd.DataFrame({"Foo": [1, 2], "Bar": [3, 4]})
        with pytest.raises(DocumentProcessingError):
            RequirementExtractionAgent().run(df, source_doc="reqs.xlsx")


# ==============================================================================
# Unit tests — Validation Agent
# ==============================================================================

def _mk_chunk(text, vendor="Acme Corp", doc_type=DocType.PROPOSAL):
    return Chunk(
        chunk_id="c1", doc_id="d1", vendor=vendor, doc_type=doc_type,
        filename="doc.pdf", text=text, position=0,
    )


class TestValidationAgent:
    def test_answer_with_valid_citation_passes(self):
        results = [(_mk_chunk("Acme Corp offers annual pricing of $50,000."), 0.9)]
        answer = "Acme Corp's annual price is $50,000 [S1]."
        report = ValidationAgent().run(answer, results)
        assert report["passed"] is True
        assert report["invalid_citations"] == []
        assert report["confidence"] in ("high", "medium")

    def test_answer_with_no_citations_flagged(self):
        results = [(_mk_chunk("Acme Corp offers annual pricing of $50,000."), 0.9)]
        answer = "Acme Corp's annual price is $50,000."
        report = ValidationAgent().run(answer, results)
        assert report["passed"] is False
        assert any("no citations" in f.lower() for f in report["flags"])

    def test_answer_with_invalid_citation_index_flagged(self):
        results = [(_mk_chunk("Some evidence text."), 0.9)]
        answer = "This claim is supported [S3]."  # only S1 exists
        report = ValidationAgent().run(answer, results)
        assert report["passed"] is False
        assert 3 in report["invalid_citations"]

    def test_no_evidence_available_is_unanswerable_but_not_failed(self):
        answer = "I could not find any relevant evidence in the uploaded documents."
        report = ValidationAgent().run(answer, [])
        assert report["snippets_available"] == 0
        assert report["confidence"] == "unanswerable"
        assert report["passed"] is True  # correctly declined to answer


# ==============================================================================
# Unit tests — Comparison / Reasoning Agent (extractive fallback, no LLM)
# ==============================================================================

class TestComparisonReasoningAgentFallback:
    def test_falls_back_to_extractive_summary_without_llm(self, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", "")  # ensure no key configured
        agent = ComparisonReasoningAgent(LLMClient())
        results = [(_mk_chunk("Vendor pricing is $10,000/year."), 0.8)]
        answer = agent.run("What is the annual price?", results)
        assert "LLM not configured" in answer
        assert "$10,000" in answer

    def test_no_results_returns_explicit_no_evidence_message(self):
        agent = ComparisonReasoningAgent(LLMClient())
        answer = agent.run("What is the annual price?", [])
        assert "could not find any relevant evidence" in answer.lower()


# ==============================================================================
# API tests — health / vendors / requirements
# ==============================================================================

class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "documents_indexed" in body


class TestVendorsAndRequirementsEndpoints:
    def test_vendors_empty_initially(self, client):
        resp = client.get("/vendors")
        assert resp.status_code == 200
        assert resp.json()["vendors"] == []

    def test_requirements_empty_initially(self, client):
        resp = client.get("/requirements")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ==============================================================================
# API tests — document upload
# ==============================================================================

class TestUploadEndpoint:
    def test_upload_csv_pricing(self, client):
        csv_bytes = make_csv_bytes([
            {"Item": "License", "AnnualCost": 12000},
            {"Item": "Support", "AnnualCost": 3000},
        ])
        resp = client.post(
            "/documents/upload",
            files={"file": ("pricing.csv", csv_bytes, "text/csv")},
            data={"vendor": "Acme Corp", "doc_type": "pricing"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["vendor"] == "Acme Corp"
        assert body["chunks_created"] >= 1
        assert body["requirements_extracted"] is None

    def test_upload_requirements_excel_extracts_requirements(self, client):
        xlsx_bytes = make_excel_bytes([
            {"Requirement ID": "REQ-001", "Description": "Must support SSO",
             "Mandatory (Y/N)": "Y", "Category": "Security"},
            {"Requirement ID": "REQ-002", "Description": "Must provide SLA >= 99.9%",
             "Mandatory (Y/N)": "Y", "Category": "Support"},
        ])
        resp = client.post(
            "/documents/upload",
            files={"file": ("requirements.xlsx", xlsx_bytes,
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"vendor": "N/A", "doc_type": "requirements"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["requirements_extracted"] == 2

        # confirm they're now retrievable via /requirements
        req_resp = client.get("/requirements")
        assert req_resp.json()["count"] == 2

    def test_upload_txt_contract(self, client):
        text = b"This agreement includes an auto-renewal clause effective annually."
        resp = client.post(
            "/documents/upload",
            files={"file": ("contract.txt", text, "text/plain")},
            data={"vendor": "Beta Inc", "doc_type": "contract"},
        )
        assert resp.status_code == 200
        assert resp.json()["chunks_created"] >= 1

    def test_upload_pdf_proposal(self, client, monkeypatch):
        # Avoid needing a real PDF binary / extra dependency — patch the loader.
        monkeypatch.setattr(app_module, "load_pdf", lambda raw: "Vendor proposal text content.")
        resp = client.post(
            "/documents/upload",
            files={"file": ("proposal.pdf", b"%PDF-fake-bytes", "application/pdf")},
            data={"vendor": "Acme Corp", "doc_type": "proposal"},
        )
        assert resp.status_code == 200
        assert resp.json()["chunks_created"] >= 1

    def test_upload_rejects_unsupported_extension(self, client):
        resp = client.post(
            "/documents/upload",
            files={"file": ("notes.docx", b"irrelevant", "application/octet-stream")},
            data={"vendor": "Acme Corp", "doc_type": "proposal"},
        )
        assert resp.status_code == 400

    def test_upload_rejects_oversized_file(self, client, monkeypatch):
        monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0)  # force any file to exceed the limit
        resp = client.post(
            "/documents/upload",
            files={"file": ("pricing.csv", b"a,b\n1,2\n", "text/csv")},
            data={"vendor": "Acme Corp", "doc_type": "pricing"},
        )
        assert resp.status_code == 413

    def test_upload_rejects_empty_extractable_text(self, client):
        resp = client.post(
            "/documents/upload",
            files={"file": ("empty.txt", b"   \n\n  ", "text/plain")},
            data={"vendor": "Acme Corp", "doc_type": "contract"},
        )
        assert resp.status_code == 422


# ==============================================================================
# API tests — /chat (retrieval + reasoning + validation, end-to-end)
# ==============================================================================

class TestChatEndpoint:
    def test_chat_rejects_empty_query(self, client):
        resp = client.post("/chat", json={"query": "   "})
        assert resp.status_code == 400

    def test_chat_with_no_documents_returns_no_evidence_answer(self, client):
        resp = client.post("/chat", json={"query": "Which vendor is cheapest?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "could not find any relevant evidence" in body["answer"].lower()
        assert body["citations"] == []
        assert body["validation"]["confidence"] == "unanswerable"

    def test_chat_retrieves_uploaded_evidence(self, client):
        text = b"Acme Corp's annual subscription price is $50,000 per year."
        client.post(
            "/documents/upload",
            files={"file": ("pricing.txt", text, "text/plain")},
            data={"vendor": "Acme Corp", "doc_type": "pricing"},
        )
        # querying with text identical to the chunk guarantees top similarity
        # under our deterministic hash-based fake embeddings
        resp = client.post("/chat", json={
            "query": "Acme Corp's annual subscription price is $50,000 per year."
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["citations"]) >= 1
        assert body["citations"][0]["vendor"] == "Acme Corp"

    def test_chat_with_llm_stub_produces_grounded_answer(self, client, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(
            LLMClient, "complete",
            lambda self, system, user, temperature=0.1, json_mode=False:
                "Acme Corp meets the mandatory SSO requirement [S1]."
        )
        text = b"Acme Corp's platform includes native SSO integration."
        client.post(
            "/documents/upload",
            files={"file": ("proposal.txt", text, "text/plain")},
            data={"vendor": "Acme Corp", "doc_type": "proposal"},
        )
        resp = client.post("/chat", json={
            "query": "Acme Corp's platform includes native SSO integration."
        })
        body = resp.json()
        assert "[S1]" in body["answer"]
        assert body["validation"]["passed"] is True
        assert body["validation"]["confidence"] in ("high", "medium")
        assert "⚠️" not in body["answer"]

    def test_chat_flags_invalid_citation_from_llm(self, client, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(
            LLMClient, "complete",
            lambda self, system, user, temperature=0.1, json_mode=False:
                "Beta Inc offers a 50% discount [S9]."  # S9 does not exist
        )
        text = b"Beta Inc's contract includes a standard payment schedule."
        client.post(
            "/documents/upload",
            files={"file": ("contract.txt", text, "text/plain")},
            data={"vendor": "Beta Inc", "doc_type": "contract"},
        )
        resp = client.post("/chat", json={
            "query": "Beta Inc's contract includes a standard payment schedule."
        })
        body = resp.json()
        assert body["validation"]["passed"] is False
        assert 9 in body["validation"]["invalid_citations"]
        assert "⚠️ Validation notice" in body["answer"]

    def test_chat_vendor_filter_excludes_other_vendors(self, client):
        client.post(
            "/documents/upload",
            files={"file": ("a.txt", b"Acme Corp mandatory feature list.", "text/plain")},
            data={"vendor": "Acme Corp", "doc_type": "proposal"},
        )
        client.post(
            "/documents/upload",
            files={"file": ("b.txt", b"Beta Inc mandatory feature list.", "text/plain")},
            data={"vendor": "Beta Inc", "doc_type": "proposal"},
        )
        resp = client.post("/chat", json={
            "query": "mandatory feature list",
            "vendor_filter": ["Acme Corp"],
        })
        body = resp.json()
        vendors_cited = {c["vendor"] for c in body["citations"]}
        assert vendors_cited.issubset({"Acme Corp"})

    def test_chat_intent_classification_pricing(self, client):
        resp = client.post("/chat", json={"query": "Compare annual pricing across vendors."})
        assert resp.json()["intent"] == "pricing_comparison"

    def test_chat_intent_classification_requirements(self, client):
        resp = client.post("/chat", json={"query": "Which vendor satisfies all mandatory requirements?"})
        assert resp.json()["intent"] == "requirements_check"

    def test_chat_intent_classification_contract(self, client):
        resp = client.post("/chat", json={"query": "Which contract contains an auto-renewal clause?"})
        assert resp.json()["intent"] == "contract_analysis"

    def test_chat_intent_classification_gap_analysis(self, client):
        resp = client.post("/chat", json={"query": "Identify missing information in the proposals."})
        assert resp.json()["intent"] == "gap_analysis"


# ==============================================================================
# API tests — reset
# ==============================================================================

class TestResetEndpoint:
    def test_reset_clears_documents_and_requirements(self, client):
        client.post(
            "/documents/upload",
            files={"file": ("a.txt", b"Some contract text.", "text/plain")},
            data={"vendor": "Acme Corp", "doc_type": "contract"},
        )
        assert client.get("/vendors").json()["vendors"] == ["Acme Corp"]

        resp = client.delete("/reset")
        assert resp.status_code == 200
        assert client.get("/vendors").json()["vendors"] == []
        assert client.get("/requirements").json()["count"] == 0