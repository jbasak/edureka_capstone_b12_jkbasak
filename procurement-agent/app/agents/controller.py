"""
app/agents/controller.py

Agent Controller
-----------------
Orchestrates the full multi-agent reasoning pipeline for each /chat request.

Pipeline
--------
ChatRequest
  │
  ├─ 1. classify_mode()       → "extractive" | "comparative"
  │
  ├─ 2. RetrievalAgent        → list[RetrievedChunk]
  │        (empty? → return NOT_FOUND_SENTINEL immediately)
  │
  ├─ 3. ReasoningAgent        → draft_answer (str)
  │
  ├─ 4. ValidationAgent       → (final_answer, citations_count, validation_passed)
  │
  └─ 5. Build ChatResponse

Mode detection
--------------
"comparative" is triggered when the question contains comparison keywords such as
"compare", "vs", "versus", "difference", "all vendors", "which vendor", "contrast",
"better", "rank".  Everything else defaults to "extractive".
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import get_settings
from app.schemas.chat import ChatRequest, ChatResponse, SourceReference
from app.agents.retrieval import RetrievalAgent, RetrievedChunk
from app.agents.reasoning import ReasoningAgent, NOT_FOUND_SENTINEL
from app.agents.validator import ValidationAgent

logger = logging.getLogger(__name__)
settings = get_settings()

# Keywords that trigger comparative mode
_COMPARATIVE_KEYWORDS = re.compile(
    r"\b(compare|comparison|vs\.?|versus|contrast|difference|differences|"
    r"all vendors|which vendor|rank|ranking|better|best|cheaper|lowest|highest|"
    r"side.by.side|across vendors)\b",
    re.IGNORECASE,
)


class AgentController:
    """
    Top-level orchestrator.  Instantiates agents on construction so they
    share the same LLM client and Qdrant connection for the request lifetime.
    """

    def __init__(
        self,
        retrieval_agent: Optional[RetrievalAgent] = None,
        reasoning_agent: Optional[ReasoningAgent] = None,
        validation_agent: Optional[ValidationAgent] = None,
    ):
        self._retrieval = retrieval_agent or RetrievalAgent()
        self._reasoning = reasoning_agent or ReasoningAgent()
        self._validation = validation_agent or ValidationAgent()

    # ── Public entry point ────────────────────────────────────────────────────

    def handle(self, request: ChatRequest) -> ChatResponse:
        """
        Execute the full agent pipeline for a single chat request.

        Parameters
        ----------
        request : validated ChatRequest from the API layer

        Returns
        -------
        ChatResponse with grounded answer, sources, and validation metadata.
        """
        question = request.question
        vendor_filter = request.vendor_filter or None

        # 1. Classify query mode
        mode = request.mode or classify_mode(question)
        logger.info(
            "AgentController: question=%r, mode=%s, vendor_filter=%s",
            question[:80],
            mode,
            vendor_filter,
        )

        # 2. Retrieve relevant chunks
        chunks: list[RetrievedChunk] = self._retrieval.retrieve(
            question=question,
            vendor_filter=vendor_filter,
        )

        # 3. Short-circuit when nothing is indexed / matched
        if not chunks:
            logger.info("AgentController: no chunks found — returning sentinel.")
            return ChatResponse(
                answer=NOT_FOUND_SENTINEL,
                mode=mode,
                sources=[],
                validation_passed=True,
                citations_count=0,
            )

        # 4. Generate answer from Groq LLM
        draft_answer = self._reasoning.reason(
            question=question,
            chunks=chunks,
            mode=mode,
        )

        # 5. Validate citations
        final_answer, citations_count, validation_passed = self._validation.validate(
            answer=draft_answer,
            chunks=chunks,
            reasoning_agent=self._reasoning,
            question=question,
            mode=mode,
        )

        # 6. Build source references for the response
        sources = _build_sources(chunks)

        logger.info(
            "AgentController: done — validation_passed=%s, citations=%d, sources=%d",
            validation_passed,
            citations_count,
            len(sources),
        )

        return ChatResponse(
            answer=final_answer,
            mode=mode,
            sources=sources,
            validation_passed=validation_passed,
            citations_count=citations_count,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_mode(question: str) -> str:
    """
    Return "comparative" if the question contains comparison keywords,
    otherwise "extractive".
    """
    if _COMPARATIVE_KEYWORDS.search(question):
        return "comparative"
    return "extractive"


def _build_sources(chunks: list[RetrievedChunk]) -> list[SourceReference]:
    """Convert RetrievedChunk objects into SourceReference Pydantic models."""
    return [
        SourceReference(
            source_file=c.source_file,
            vendor_name=c.vendor_name,
            doc_category=c.doc_category,
            page_number=c.page_number,
            row_index=c.row_index,
            sheet_name=c.sheet_name,
            chunk_index=c.chunk_index,
            score=c.score,
            text_excerpt=c.short_excerpt(200),
        )
        for c in chunks
    ]
