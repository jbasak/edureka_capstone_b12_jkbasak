"""
app/api/chat.py

POST /chat      — ask a natural-language question across indexed documents
POST /chat/compliance — generate a compliance matrix for a requirements file

Both endpoints delegate all reasoning to the AgentController.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas.chat import ChatRequest, ChatResponse
from app.agents.controller import AgentController

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared controller instance (agents are cheap to create; controller is
# stateless so one instance per process is fine)
_controller = AgentController()


# ── POST /chat ────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a procurement question",
    description=(
        "Submit a natural-language question. The multi-agent pipeline "
        "retrieves relevant document chunks from ChromaDB, reasons over them "
        "using the Groq LLM, and validates that every claim is cited. "
        "Returns 'Information not found in context.' when evidence is absent."
    ),
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Full agent pipeline:
      classify → retrieve → reason (Groq LLM) → validate → respond
    """
    logger.info(
        "POST /chat  question=%r  vendor_filter=%s  mode=%s",
        request.question[:80],
        request.vendor_filter,
        request.mode,
    )

    try:
        response = _controller.handle(request)
    except RuntimeError as exc:
        # LLM / ChromaDB failures surfaced as 502 so the caller can distinguish
        # application errors from 4xx validation errors
        logger.exception("Agent pipeline failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent pipeline error: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error in /chat")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        )

    return response


# ── POST /chat/compliance ─────────────────────────────────────────────────────

@router.post(
    "/compliance",
    summary="Generate a vendor compliance matrix",
    description=(
        "Upload an Excel or CSV requirements checklist. "
        "For each requirement row, the system evaluates whether each specified "
        "vendor satisfies it (Pass / Fail / Partially Met / Not Found) and "
        "returns a Markdown compliance table with citations."
    ),
)
async def compliance_matrix(
    file: UploadFile = File(..., description="Requirements checklist (.xlsx or .csv)"),
    vendors: str = Form(
        ...,
        description="Comma-separated vendor names to evaluate, e.g. 'AWS,Google,Oracle'",
    ),
) -> dict:
    """
    Generates a compliance matrix for the uploaded requirements file
    against the specified vendors.
    """
    filename = file.filename or "requirements.xlsx"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ("xlsx", "csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Compliance endpoint accepts .xlsx or .csv files only.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded requirements file is empty.",
        )

    vendor_list = [v.strip() for v in vendors.split(",") if v.strip()]
    if not vendor_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'vendors' field must contain at least one vendor name.",
        )

    logger.info(
        "POST /chat/compliance  file='%s', vendors=%s",
        filename,
        vendor_list,
    )

    # Import here to avoid circular imports at module level
    from app.agents.extractor import RequirementExtractionAgent
    from app.agents.retrieval import RetrievalAgent
    from app.agents.reasoning import ReasoningAgent

    extractor = RequirementExtractionAgent()
    retrieval = RetrievalAgent()
    reasoning = ReasoningAgent()

    try:
        requirements = extractor.extract_requirements(
            file_bytes=file_bytes,
            filename=filename,
        )
    except Exception as exc:
        logger.exception("Requirement extraction failed for '%s'", filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract requirements: {exc}",
        )

    if not requirements:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No requirements found in '{filename}'.",
        )

    try:
        verdicts = extractor.build_compliance_matrix(
            requirements=requirements,
            vendor_names=vendor_list,
            retrieval_agent=retrieval,
            reasoning_agent=reasoning,
        )
    except RuntimeError as exc:
        logger.exception("Compliance matrix generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM error during compliance evaluation: {exc}",
        )

    table_md = extractor.format_compliance_table(
        requirements=requirements,
        verdicts=verdicts,
        vendor_names=vendor_list,
    )

    return {
        "status": "success",
        "requirements_count": len(requirements),
        "vendors": vendor_list,
        "compliance_table": table_md,
        "verdicts": [
            {
                "requirement_id": v.requirement_id,
                "vendor_name": v.vendor_name,
                "status": v.status,
                "evidence": v.evidence,
                "citation": v.citation,
            }
            for v in verdicts
        ],
    }
