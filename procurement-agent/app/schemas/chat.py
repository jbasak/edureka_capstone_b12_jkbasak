"""
app/schemas/chat.py

Pydantic v2 models for the /chat request and response.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """POST /chat request body."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        examples=["Compare annual pricing across AWS, Google, and Oracle"],
    )
    vendor_filter: Optional[List[str]] = Field(
        None,
        description="Limit retrieval to these vendor names (case-sensitive).",
        examples=[["AWS", "Google", "Oracle"]],
    )
    mode: Optional[Literal["extractive", "comparative"]] = Field(
        None,
        description=(
            "Query mode. 'extractive' for single-vendor fact lookup; "
            "'comparative' for cross-vendor analysis. "
            "Auto-detected from question keywords if omitted."
        ),
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class SourceReference(BaseModel):
    """A single retrieved chunk that supports a factual claim in the answer."""

    source_file: str
    vendor_name: Optional[str] = None
    doc_category: Optional[str] = None
    page_number: Optional[int] = None
    row_index: Optional[int] = None
    sheet_name: Optional[str] = None
    chunk_index: int
    score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    text_excerpt: Optional[str] = Field(
        None,
        description="First 200 characters of the chunk text for quick preview",
    )


class ChatResponse(BaseModel):
    """POST /chat response body."""

    answer: str = Field(
        ...,
        description=(
            "Grounded answer from the LLM. Contains Markdown formatting for "
            "comparative responses. Returns 'Information not found in context.' "
            "when no relevant evidence exists."
        ),
    )
    mode: str = Field(..., examples=["comparative"])
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Retrieved chunks that were passed to the LLM as context.",
    )
    validation_passed: bool = Field(
        ...,
        description="True if the Validation Agent confirmed all claims are cited.",
    )
    citations_count: int = Field(
        ...,
        ge=0,
        description="Number of bracketed citations found in the answer text.",
    )
