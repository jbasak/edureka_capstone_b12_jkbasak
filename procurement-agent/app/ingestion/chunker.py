"""
app/ingestion/chunker.py

Token-aware text chunker with configurable size and overlap.

Strategy
--------
1. Split the raw text into sentences using a simple regex splitter.
2. Accumulate sentences into a window until the token count reaches CHUNK_SIZE.
3. When the window is full, emit a TextChunk, then slide back CHUNK_OVERLAP
   tokens worth of sentences to start the next window (overlap).
4. Small documents that fit within a single window produce exactly one chunk.

Token counting uses tiktoken (cl100k_base encoding — compatible with most
modern LLMs).  Falls back to a whitespace-word count if tiktoken is unavailable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.ingestion.parser import ParsedChunk

logger = logging.getLogger(__name__)

# ── Encoding ──────────────────────────────────────────────────────────────────
try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))

except ImportError:
    logger.warning("tiktoken not installed; falling back to whitespace token count.")

    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text.split())


# ── Output model ──────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A token-bounded chunk ready for embedding + vector storage."""

    text: str
    source_file: str
    vendor_name: Optional[str]
    doc_category: Optional[str]
    page_number: Optional[int]
    row_index: Optional[int]
    sheet_name: Optional[str]
    chunk_index: int           # 0-based index within the parent document
    total_chunks: int          # filled in after all chunks are produced
    token_count: int


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+"          # end-of-sentence punctuation followed by whitespace
    r"|(?<=\n)\s*\n"          # blank lines (paragraph breaks)
    r"|\n(?=[A-Z0-9])"        # newline before a capital / digit (heading hint)
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like fragments; filter empty results."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Core chunker ──────────────────────────────────────────────────────────────

def chunk_parsed_document(
    parsed_chunks: list[ParsedChunk],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """
    Convert a list of ParsedChunk objects (one per page / row) into
    token-bounded TextChunk objects suitable for embedding.

    Parameters
    ----------
    parsed_chunks  : output of parser.parse_document()
    chunk_size     : maximum tokens per chunk
    chunk_overlap  : token overlap between consecutive chunks

    Returns
    -------
    List of TextChunk with chunk_index and total_chunks filled in.
    """
    all_chunks: list[TextChunk] = []

    for parsed in parsed_chunks:
        page_chunks = _chunk_text(
            text=parsed.text,
            source_file=parsed.source_file,
            vendor_name=parsed.vendor_name,
            doc_category=parsed.doc_category,
            page_number=parsed.page_number,
            row_index=parsed.row_index,
            sheet_name=parsed.sheet_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_index=len(all_chunks),
        )
        all_chunks.extend(page_chunks)

    # Back-fill total_chunks now that we know the final count
    total = len(all_chunks)
    for chunk in all_chunks:
        chunk.total_chunks = total

    logger.debug(
        "Chunked '%s' → %d TextChunks (size=%d, overlap=%d)",
        all_chunks[0].source_file if all_chunks else "?",
        total,
        chunk_size,
        chunk_overlap,
    )
    return all_chunks


def _chunk_text(
    text: str,
    source_file: str,
    vendor_name: Optional[str],
    doc_category: Optional[str],
    page_number: Optional[int],
    row_index: Optional[int],
    sheet_name: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    base_index: int,
) -> list[TextChunk]:
    """
    Sliding-window sentence-level chunker for a single text block.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[TextChunk] = []
    window: list[str] = []
    window_tokens: int = 0
    local_index: int = 0

    def emit(win: list[str]) -> TextChunk:
        nonlocal local_index
        joined = " ".join(win)
        tc = TextChunk(
            text=joined,
            source_file=source_file,
            vendor_name=vendor_name,
            doc_category=doc_category,
            page_number=page_number,
            row_index=row_index,
            sheet_name=sheet_name,
            chunk_index=base_index + local_index,
            total_chunks=0,  # filled later
            token_count=_count_tokens(joined),
        )
        local_index += 1
        return tc

    for sentence in sentences:
        sent_tokens = _count_tokens(sentence)

        # A single sentence that exceeds chunk_size is split hard
        if sent_tokens > chunk_size:
            words = sentence.split()
            sub_buf: list[str] = []
            sub_tokens = 0
            for word in words:
                wt = _count_tokens(word)
                if sub_tokens + wt > chunk_size and sub_buf:
                    if window:
                        window.extend(sub_buf)
                        chunks.append(emit(window))
                        window, window_tokens = _trim_overlap(window, chunk_overlap)
                        sub_buf = []
                        sub_tokens = 0
                    else:
                        chunks.append(emit(sub_buf))
                        sub_buf = []
                        sub_tokens = 0
                sub_buf.append(word)
                sub_tokens += wt
            if sub_buf:
                window.extend(sub_buf)
                window_tokens += sub_tokens
            continue

        if window_tokens + sent_tokens > chunk_size and window:
            chunks.append(emit(window))
            window, window_tokens = _trim_overlap(window, chunk_overlap)

        window.append(sentence)
        window_tokens += sent_tokens

    # Emit any remaining content
    if window:
        chunks.append(emit(window))

    return chunks


def _trim_overlap(window: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """
    Return the suffix of *window* whose total token count is ≤ overlap_tokens.
    This forms the overlap carried into the next chunk.
    """
    kept: list[str] = []
    kept_tokens = 0
    for sentence in reversed(window):
        st = _count_tokens(sentence)
        if kept_tokens + st <= overlap_tokens:
            kept.insert(0, sentence)
            kept_tokens += st
        else:
            break
    return kept, kept_tokens
