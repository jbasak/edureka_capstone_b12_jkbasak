"""
app/agents/reasoning.py

Comparison & Reasoning Agent
------------------------------
Calls the Groq LLM (via OpenAI-compatible SDK) with a strict system prompt
that forces grounded, cited answers.

Key behaviours
--------------
- Uses `openai.OpenAI` with `base_url` pointed at Groq's API.
- System prompt instructs the model to:
    * Answer ONLY from the provided context chunks.
    * Return the sentinel "Information not found in context." when evidence is absent.
    * Include bracketed citations `[filename, Page N]` for every factual claim.
    * Format comparative answers as Markdown tables.
- `reason()` assembles a structured context block from RetrievedChunk objects
  and passes it as a user-turn message alongside the question.
- `_build_context_block()` renders each chunk with its citation label so the
  model can reference them naturally.
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from app.config import get_settings
from app.agents.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Sentinel returned when no relevant context is found ──────────────────────
NOT_FOUND_SENTINEL = "Information not found in context."

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a precise procurement analysis assistant helping evaluate \
vendor responses to a government RFP (Request for Proposal).

Rules you MUST follow without exception:
1. Answer ONLY using information from the CONTEXT CHUNKS provided below.
2. If the context does not contain sufficient information to answer, respond with \
exactly: "Information not found in context."
3. Every factual claim in your answer MUST include a bracketed citation in the \
format [filename, Page N] or [filename, Row N] referencing the source chunk.
4. Do NOT infer, extrapolate, or fabricate any information not present in the context.
5. For comparative questions (comparing multiple vendors), produce a Markdown table.
6. For extractive questions (single vendor or single fact), produce a clear prose answer.
7. Keep answers concise and professional. Use bullet points where helpful.
"""


class ReasoningAgent:
    """
    Sends retrieved chunks + question to Groq LLM and returns a grounded answer.
    """

    def __init__(self, llm_client: Optional[OpenAI] = None):
        self._client = llm_client or OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        self._model = settings.groq_model

    def reason(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        mode: str = "extractive",
    ) -> str:
        """
        Generate a grounded answer from retrieved chunks.

        Parameters
        ----------
        question : the user's natural-language question
        chunks   : list of RetrievedChunk objects from the Retrieval Agent
        mode     : "extractive" or "comparative" — hints the model's output format

        Returns
        -------
        LLM-generated answer string (Markdown). Returns NOT_FOUND_SENTINEL if
        chunks is empty or the model cannot find evidence.
        """
        if not chunks:
            logger.info("ReasoningAgent: no chunks provided — returning sentinel.")
            return NOT_FOUND_SENTINEL

        context_block = _build_context_block(chunks)
        mode_hint = (
            "Produce a Markdown comparison table." if mode == "comparative"
            else "Provide a precise prose answer."
        )

        user_message = (
            f"CONTEXT CHUNKS:\n{context_block}\n\n"
            f"QUESTION: {question}\n\n"
            f"INSTRUCTION: {mode_hint} Include bracketed citations for every claim."
        )

        logger.info(
            "ReasoningAgent: calling Groq model='%s', mode='%s', chunks=%d",
            self._model,
            mode,
            len(chunks),
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,     # low temperature → factual, deterministic
                max_tokens=2048,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("ReasoningAgent: LLM call failed: %s", exc)
            raise RuntimeError(f"LLM call failed: {exc}") from exc

        # Guard: if the model returns empty string fall back to sentinel
        if not answer:
            return NOT_FOUND_SENTINEL

        logger.info(
            "ReasoningAgent: answer generated (%d chars).", len(answer)
        )
        return answer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Render retrieved chunks as a numbered context block for the LLM prompt.

    Format
    ------
    [1] Source: filename | Page 3 | Vendor: AWS | Score: 0.91
    <chunk text>

    [2] Source: filename | Row 7 | Vendor: Oracle | Score: 0.88
    <chunk text>
    ...
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        # Build location string
        if chunk.page_number is not None:
            location = f"Page {chunk.page_number}"
        elif chunk.row_index is not None:
            location = f"Row {chunk.row_index}"
        else:
            location = f"Chunk {chunk.chunk_index}"

        vendor_str = f" | Vendor: {chunk.vendor_name}" if chunk.vendor_name else ""
        header = (
            f"[{i}] Source: {chunk.source_file} | {location}"
            f"{vendor_str} | Score: {chunk.score:.2f}"
        )
        lines.append(header)
        lines.append(chunk.text.strip())
        lines.append("")  # blank line separator

    return "\n".join(lines)
