"""
app/agents/validator.py

Validation Agent
-----------------
Verifies that the Reasoning Agent's output is fully grounded — i.e. every
substantive factual claim carries at least one bracketed citation that maps
back to a source file present in the retrieved chunks.

Algorithm
---------
1. Extract all bracketed citations from the answer text using regex.
2. Build a set of "known" source files from the retrieved chunks.
3. Check that every citation references a known source file.
4. Check that every substantive sentence (length > 20 chars, not a heading or
   bullet marker) is accompanied by at least one citation in its paragraph.
5. If validation fails, allow ONE rewrite attempt by calling the Reasoning Agent
   with an explicit correction instruction.
6. Return (final_text, citations_count, validation_passed).

Design notes
------------
- Citation detection is intentionally lenient: the regex accepts both
  [filename.pdf, Page 3] and [filename, Row 7] and partial matches.
- The rewrite prompt appends a targeted correction rather than repeating the
  full context, to save tokens.
- "Information not found in context." is always considered valid (no citations
  required for the sentinel string).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.agents.retrieval import RetrievedChunk
from app.agents.reasoning import NOT_FOUND_SENTINEL

logger = logging.getLogger(__name__)

# Matches [any text, Page N] or [any text, Row N] or [any text, Chunk N]
_CITATION_RE = re.compile(
    r"\[([^\]]+?),\s*(?:Page|Row|Chunk)\s+\d+\]",
    re.IGNORECASE,
)

# Lines that are exempt from the citation requirement
_EXEMPT_LINE_RE = re.compile(
    r"^\s*(?:#|\|[-| ]+\||\*\*|>|[-*•–]|\|)",  # headings, table dividers, bullets
    re.MULTILINE,
)


class ValidationAgent:
    """
    Checks citation completeness and triggers one rewrite if needed.
    """

    def validate(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        reasoning_agent=None,         # Optional[ReasoningAgent] — for rewrite
        question: Optional[str] = None,
        mode: str = "extractive",
    ) -> tuple[str, int, bool]:
        """
        Validate the answer against retrieved chunks.

        Parameters
        ----------
        answer           : raw LLM answer string
        chunks           : list of RetrievedChunk objects used as context
        reasoning_agent  : if provided, used for a single rewrite attempt
        question         : original user question (needed for rewrite)
        mode             : query mode passed to reasoning_agent on rewrite

        Returns
        -------
        (final_answer, citations_count, validation_passed)
        """
        # Sentinel is always valid
        if answer.strip() == NOT_FOUND_SENTINEL:
            return answer, 0, True

        citations = _CITATION_RE.findall(answer)
        citations_count = len(citations)

        known_sources = {c.source_file for c in chunks}
        passed, issues = self._check(answer, citations, known_sources)

        if passed:
            logger.info(
                "ValidationAgent: PASSED (%d citations, %d chunks).",
                citations_count,
                len(chunks),
            )
            return answer, citations_count, True

        logger.warning(
            "ValidationAgent: FAILED — issues: %s. Attempting rewrite…", issues
        )

        # ── Single rewrite attempt ────────────────────────────────────────────
        if reasoning_agent is not None and question is not None:
            rewrite_hint = (
                f"Your previous answer had the following citation issues:\n"
                f"{chr(10).join(issues)}\n\n"
                f"Please rewrite the answer ensuring EVERY factual sentence "
                f"includes a bracketed citation in the format "
                f"[filename, Page N] or [filename, Row N]."
            )
            try:
                # Inject correction as an additional user instruction
                from app.agents.reasoning import _build_context_block
                context_block = _build_context_block(chunks)
                rewrite_question = (
                    f"{question}\n\n[CORRECTION REQUIRED]\n{rewrite_hint}"
                )

                rewritten = reasoning_agent.reason(
                    question=rewrite_question,
                    chunks=chunks,
                    mode=mode,
                )

                # Re-validate rewritten answer (no further retries)
                re_citations = _CITATION_RE.findall(rewritten)
                re_count = len(re_citations)
                re_passed, _ = self._check(rewritten, re_citations, known_sources)

                logger.info(
                    "ValidationAgent: rewrite result — passed=%s, citations=%d.",
                    re_passed,
                    re_count,
                )
                return rewritten, re_count, re_passed

            except Exception as exc:
                logger.error("ValidationAgent: rewrite failed: %s", exc)
                # Return original answer with failed flag rather than crashing
                return answer, citations_count, False

        return answer, citations_count, False

    # ── Internal checks ───────────────────────────────────────────────────────

    def _check(
        self,
        answer: str,
        citations: list[str],
        known_sources: set[str],
    ) -> tuple[bool, list[str]]:
        """
        Returns (passed, list_of_issue_strings).
        """
        issues: list[str] = []

        # 1. Every citation must reference a known source file
        for citation_text in citations:
            # citation_text is the content inside the brackets, e.g.
            # "AWS_Proposal.pdf, Page 3" — check if any known source is a
            # substring of the citation (handles filename variations)
            if not any(
                src.lower() in citation_text.lower() or
                citation_text.lower() in src.lower()
                for src in known_sources
            ):
                issues.append(
                    f"Citation '[{citation_text}]' does not match any known source file."
                )

        # 2. Every substantive line must be covered by at least one citation
        #    We check per-paragraph: a paragraph passes if it contains ≥1 citation
        paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
        for para in paragraphs:
            if _is_exempt(para):
                continue
            substantive_lines = [
                ln for ln in para.splitlines()
                if len(ln.strip()) > 20 and not _is_exempt(ln)
            ]
            if not substantive_lines:
                continue
            if not _CITATION_RE.search(para):
                # Truncate for readability in the log/issue message
                preview = para[:80].replace("\n", " ")
                issues.append(
                    f"Paragraph has no citation: \"{preview}…\""
                )

        passed = len(issues) == 0
        return passed, issues


def _is_exempt(line: str) -> bool:
    """Return True if the line is a heading, table row, bullet, or divider."""
    stripped = line.strip()
    if not stripped:
        return True
    return bool(_EXEMPT_LINE_RE.match(line))
