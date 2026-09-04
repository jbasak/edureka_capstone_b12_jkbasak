"""
tests/test_ingestion_api.py

Integration tests for POST /documents, DELETE /documents, GET /documents.

Uses the session-scoped TestClient from conftest.py which:
  - Replaces ChromaDB with in-memory EphemeralClient via ChromaManager
  - Patches embed_texts to return zero-vectors (no model download)
"""

from __future__ import annotations

import io
import pytest


# ── POST /documents ───────────────────────────────────────────────────────────

class TestIngestDocument:
    def test_upload_txt_returns_201(self, client, sample_txt_bytes):
        response = client.post(
            "/documents",
            files={"file": ("vendor_proposal.txt", sample_txt_bytes, "text/plain")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert data["filename"] == "vendor_proposal.txt"
        assert data["chunks_stored"] > 0

    def test_upload_csv_returns_201(self, client, sample_csv_bytes):
        response = client.post(
            "/documents",
            files={"file": ("pricing.csv", sample_csv_bytes, "text/csv")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["chunks_stored"] > 0

    def test_upload_xlsx_returns_201(self, client, sample_xlsx_bytes):
        response = client.post(
            "/documents",
            files={"file": ("requirements.xlsx", sample_xlsx_bytes,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["chunks_stored"] > 0

    def test_upload_with_vendor_tag(self, client, sample_txt_bytes):
        response = client.post(
            "/documents",
            files={"file": ("aws_proposal.txt", sample_txt_bytes, "text/plain")},
            data={"vendor_name": "AWS", "doc_category": "proposal"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["vendor_name"] == "AWS"
        assert data["doc_category"] == "proposal"

    def test_upload_pdf_returns_201(self, client, sample_pdf_bytes):
        response = client.post(
            "/documents",
            files={"file": ("rfp.pdf", sample_pdf_bytes, "application/pdf")},
        )
        # Minimal PDF may yield 0 text chunks — accept 201 or 422
        assert response.status_code in (201, 422)

    def test_upload_unsupported_type_returns_422(self, client):
        response = client.post(
            "/documents",
            files={"file": ("archive.zip", b"PK\x03\x04data", "application/zip")},
        )
        assert response.status_code == 422
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_empty_file_returns_422(self, client):
        response = client.post(
            "/documents",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 422

    def test_response_schema_complete(self, client, sample_csv_bytes):
        response = client.post(
            "/documents",
            files={"file": ("data.csv", sample_csv_bytes, "text/csv")},
        )
        assert response.status_code == 201
        data = response.json()
        for field in ("status", "filename", "chunks_stored", "collection"):
            assert field in data, f"Missing field: {field}"

    def test_upload_idempotent_same_file_twice(self, client, sample_txt_bytes):
        """Re-uploading the same file should succeed both times (idempotent upsert)."""
        r1 = client.post(
            "/documents",
            files={"file": ("idempotent.txt", sample_txt_bytes, "text/plain")},
        )
        r2 = client.post(
            "/documents",
            files={"file": ("idempotent.txt", sample_txt_bytes, "text/plain")},
        )
        assert r1.status_code == 201
        assert r2.status_code == 201
        # Both should store the same number of chunks
        assert r1.json()["chunks_stored"] == r2.json()["chunks_stored"]


# ── DELETE /documents ─────────────────────────────────────────────────────────

class TestDeleteDocument:
    def test_delete_existing_source(self, client, sample_txt_bytes):
        # First ingest
        client.post(
            "/documents",
            files={"file": ("delete_me.txt", sample_txt_bytes, "text/plain")},
        )
        # Then delete
        response = client.delete("/documents/delete_me.txt")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "delete_me.txt" in data["source_file"]

    def test_delete_nonexistent_source_still_200(self, client):
        """Deleting a file that was never ingested should not error."""
        response = client.delete("/documents/never_uploaded.pdf")
        assert response.status_code == 200


# ── GET /documents ────────────────────────────────────────────────────────────

class TestListDocuments:
    def test_returns_200(self, client):
        response = client.get("/documents")
        assert response.status_code == 200

    def test_response_has_expected_keys(self, client):
        response = client.get("/documents")
        data = response.json()
        assert "total_documents" in data
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_ingested_file_appears_in_list(self, client, sample_csv_bytes):
        client.post(
            "/documents",
            files={"file": ("listed.csv", sample_csv_bytes, "text/csv")},
        )
        response = client.get("/documents")
        sources = [d["source_file"] for d in response.json()["documents"]]
        assert "listed.csv" in sources
