"""
app/api/documents.py

POST /documents  — ingest a single document into the vector store
DELETE /documents/{filename} — remove all vectors for a source file
GET  /documents  — list all indexed source files with chunk counts

Full ingestion pipeline per upload:
  UploadFile → parse_document() → chunk_parsed_document()
             → embed_texts()    → qdrant.upsert_chunks()
             → DocumentUploadResponse (201)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import get_settings
from app.ingestion.parser import parse_document, SUPPORTED_EXTENSIONS
from app.ingestion.chunker import chunk_parsed_document
from app.ingestion.embedder import embed_texts
from app.vector_store.qdrant_client import get_qdrant_manager
from app.schemas.document import DocumentUploadResponse, DeleteDocumentResponse

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

_MAX_BYTES = settings.max_upload_mb * 1024 * 1024  # MB → bytes


# ── POST /documents ───────────────────────────────────────────────────────────

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentUploadResponse,
    summary="Ingest a document",
    description=(
        "Upload a PDF, TXT, CSV, XLSX, or DOCX file. The document is parsed, "
        "chunked, embedded, and stored in Qdrant. Re-uploading the same file "
        "is idempotent — existing vectors are overwritten."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="Document to ingest"),
    vendor_name: Optional[str] = Form(
        None,
        description="Vendor label (e.g. AWS, Google, Oracle)",
    ),
    doc_category: Optional[str] = Form(
        None,
        description="Category tag (e.g. proposal, pricing, requirements, contract)",
    ),
) -> DocumentUploadResponse:
    filename = file.filename or "unknown"

    # ── Extension guard ───────────────────────────────────────────────────────
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # ── Size guard ────────────────────────────────────────────────────────────
    file_bytes = await file.read()
    if len(file_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {len(file_bytes) / (1024*1024):.1f} MB exceeds "
                f"maximum allowed {settings.max_upload_mb} MB."
            ),
        )

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    logger.info(
        "Ingesting '%s' (%.1f KB, vendor=%s, category=%s)",
        filename,
        len(file_bytes) / 1024,
        vendor_name,
        doc_category,
    )

    # ── Parse → Chunk → Embed → Store ─────────────────────────────────────────
    try:
        parsed = parse_document(
            file_bytes=file_bytes,
            filename=filename,
            vendor_name=vendor_name,
            doc_category=doc_category,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    except RuntimeError as exc:
        logger.exception("Parsing failed for '%s'", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    if not parsed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No extractable text found in '{filename}'.",
        )

    chunks = chunk_parsed_document(
        parsed_chunks=parsed,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Chunking produced no output for '{filename}'.",
        )

    texts = [c.text for c in chunks]
    try:
        vectors = embed_texts(texts)
    except Exception as exc:
        logger.exception("Embedding failed for '%s'", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {exc}",
        )

    qdrant = get_qdrant_manager()
    try:
        stored = qdrant.upsert_chunks(chunks=chunks, vectors=vectors)
    except Exception as exc:
        logger.exception("Qdrant upsert failed for '%s'", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store write failed: {exc}",
        )

    logger.info(
        "Ingestion complete: '%s' → %d chunks stored.", filename, stored
    )

    return DocumentUploadResponse(
        status="success",
        filename=filename,
        vendor_name=vendor_name,
        doc_category=doc_category,
        chunks_stored=stored,
        collection=settings.qdrant_collection,
    )


# ── DELETE /documents/{filename} ──────────────────────────────────────────────

@router.delete(
    "/{filename:path}",
    response_model=DeleteDocumentResponse,
    summary="Remove a document from the index",
    description="Delete all vector points associated with the given source filename.",
)
def delete_document(filename: str) -> DeleteDocumentResponse:
    qdrant = get_qdrant_manager()
    try:
        qdrant.delete_by_source(filename)
    except Exception as exc:
        logger.exception("Delete failed for '%s'", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {exc}",
        )
    return DeleteDocumentResponse(
        status="success",
        source_file=filename,
        message=f"All vectors for '{filename}' have been removed.",
    )


# ── GET /documents ────────────────────────────────────────────────────────────

@router.get(
    "",
    summary="List indexed documents",
    description="Returns all distinct source files currently stored in the vector collection.",
)
def list_documents() -> dict:
    """
    Scroll the entire Qdrant collection and aggregate unique source_file values
    with their chunk counts.  Practical for collections up to ~100 k points.
    """
    qdrant = get_qdrant_manager()
    try:
        from qdrant_client.http import models as qmodels

        source_counts: dict[str, int] = {}
        offset = None

        while True:
            result, next_offset = qdrant._client.scroll(
                collection_name=settings.qdrant_collection,
                limit=256,
                offset=offset,
                with_payload=["source_file", "vendor_name"],
                with_vectors=False,
            )
            for point in result:
                src = (point.payload or {}).get("source_file", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1

            if next_offset is None:
                break
            offset = next_offset

        documents = [
            {"source_file": src, "chunks": count, "vendor_name": None}
            for src, count in sorted(source_counts.items())
        ]
        return {"total_documents": len(documents), "documents": documents}

    except Exception as exc:
        logger.exception("Failed to list documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not list documents: {exc}",
        )
