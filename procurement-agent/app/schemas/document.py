"""
app/schemas/document.py

Pydantic v2 models for document ingestion request / response validation.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DocumentUploadResponse(BaseModel):
    """Returned by POST /documents on success (HTTP 201)."""

    status: str = Field(..., examples=["success"])
    filename: str = Field(..., examples=["AWS_Proposal.pdf"])
    vendor_name: Optional[str] = Field(None, examples=["AWS"])
    doc_category: Optional[str] = Field(None, examples=["proposal"])
    chunks_stored: int = Field(..., ge=0, description="Number of vector chunks committed to Qdrant")
    collection: str = Field(..., examples=["procurement_docs"])


class ChunkMetadata(BaseModel):
    """
    Metadata payload stored alongside every vector point in Qdrant.
    Also used internally to pass provenance between pipeline stages.
    """

    source_file: str
    vendor_name: Optional[str] = None
    doc_category: Optional[str] = None
    page_number: Optional[int] = Field(None, ge=1, description="1-based page number (PDF/DOCX)")
    row_index: Optional[int] = Field(None, ge=0, description="0-based row index (CSV/XLSX)")
    sheet_name: Optional[str] = None
    chunk_index: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=1)
    token_count: int = Field(..., ge=0)
    text: Optional[str] = None  # stored in Qdrant payload for citation display


class DeleteDocumentResponse(BaseModel):
    """Returned by DELETE /documents/{filename}."""

    status: str
    source_file: str
    message: str
