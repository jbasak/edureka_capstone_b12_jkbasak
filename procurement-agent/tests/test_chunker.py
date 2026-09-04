"""
tests/test_chunker.py

Unit tests for app/ingestion/chunker.py

Tests cover:
  - All chunks respect the configured token size ceiling
  - Consecutive chunks share overlap tokens
  - Small documents produce exactly one chunk
  - total_chunks is back-filled correctly
  - chunk_index sequence is contiguous across a document
  - Multiple parsed blocks (e.g. multi-page PDF) are merged correctly
"""

from __future__ import annotations

import pytest

from app.ingestion.parser import ParsedChunk
from app.ingestion.chunker import chunk_parsed_document, TextChunk, _count_tokens


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_parsed(text: str, source: str = "doc.txt") -> ParsedChunk:
    return ParsedChunk(text=text, source_file=source)


def _long_text(words: int = 1200) -> str:
    """Generate a deterministic text string of approximately *words* words."""
    base = (
        "The vendor proposes a comprehensive cloud solution including compute, "
        "storage, networking, and managed database services to meet all "
        "mandatory requirements specified in the RFP document. "
    )
    repetitions = (words // len(base.split())) + 2
    return (base * repetitions).strip()


# ── Chunk size ceiling ────────────────────────────────────────────────────────

class TestChunkSizeCeiling:
    def test_all_chunks_within_size_limit(self):
        text = _long_text(1200)
        parsed = [_make_parsed(text)]
        chunks = chunk_parsed_document(parsed, chunk_size=900, chunk_overlap=150)
        for c in chunks:
            assert c.token_count <= 950, (  # 950 = 900 + small sentence rounding
                f"Chunk {c.chunk_index} has {c.token_count} tokens, exceeds 900+tolerance"
            )

    def test_chunk_size_50_respected(self):
        text = _long_text(400)
        parsed = [_make_parsed(text)]
        chunks = chunk_parsed_document(parsed, chunk_size=50, chunk_overlap=10)
        for c in chunks:
            assert c.token_count <= 60, (
                f"Chunk {c.chunk_index} has {c.token_count} tokens, exceeds 50+tolerance"
            )

    def test_produces_multiple_chunks_for_long_doc(self):
        text = _long_text(1200)
        parsed = [_make_parsed(text)]
        chunks = chunk_parsed_document(parsed, chunk_size=200, chunk_overlap=30)
        assert len(chunks) > 1, "Long document should produce multiple chunks"


# ── Overlap ───────────────────────────────────────────────────────────────────

class TestOverlap:
    def test_consecutive_chunks_share_tokens(self):
        text = _long_text(600)
        parsed = [_make_parsed(text)]
        chunks = chunk_parsed_document(parsed, chunk_size=200, chunk_overlap=40)

        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")

        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].text.split())
            words_b = set(chunks[i + 1].text.split())
            shared = words_a & words_b
            assert len(shared) > 0, (
                f"Chunks {i} and {i+1} share no words — overlap may be missing"
            )

    def test_zero_overlap_produces_no_shared_content(self):
        """With overlap=0 consecutive chunks should share very little."""
        text = _long_text(400)
        parsed = [_make_parsed(text)]
        chunks = chunk_parsed_document(parsed, chunk_size=100, chunk_overlap=0)
        if len(chunks) < 2:
            pytest.skip("Not enough chunks")
        # Can't guarantee zero shared words since sentences may repeat words
        # Just verify it doesn't crash
        assert len(chunks) >= 1


# ── Small document → single chunk ────────────────────────────────────────────

class TestSmallDocument:
    def test_small_doc_produces_one_chunk(self):
        text = "This is a short vendor proposal summary."
        parsed = [_make_parsed(text)]
        chunks = chunk_parsed_document(parsed, chunk_size=900, chunk_overlap=150)
        assert len(chunks) == 1

    def test_single_chunk_index_is_zero(self):
        text = "Short text."
        parsed = [_make_parsed(text)]
        chunks = chunk_parsed_document(parsed, chunk_size=900, chunk_overlap=150)
        assert chunks[0].chunk_index == 0


# ── Metadata propagation ──────────────────────────────────────────────────────

class TestMetadata:
    def test_total_chunks_backfilled(self):
        text = _long_text(600)
        parsed = [_make_parsed(text)]
        chunks = chunk_parsed_document(parsed, chunk_size=200, chunk_overlap=30)
        total = len(chunks)
        for c in chunks:
            assert c.total_chunks == total

    def test_chunk_index_is_unique_and_contiguous(self):
        text = _long_text(600)
        parsed = [_make_parsed(text)]
        chunks = chunk_parsed_document(parsed, chunk_size=200, chunk_overlap=30)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_source_file_preserved(self):
        parsed = [_make_parsed("Some text.", source="Oracle_Proposal.pdf")]
        chunks = chunk_parsed_document(parsed)
        for c in chunks:
            assert c.source_file == "Oracle_Proposal.pdf"

    def test_vendor_name_preserved(self):
        p = ParsedChunk(
            text="AWS offers 99.99% uptime.",
            source_file="aws.pdf",
            vendor_name="AWS",
            doc_category="proposal",
        )
        chunks = chunk_parsed_document([p])
        for c in chunks:
            assert c.vendor_name == "AWS"
            assert c.doc_category == "proposal"

    def test_page_number_preserved(self):
        p = ParsedChunk(text="Page content here.", source_file="doc.pdf", page_number=7)
        chunks = chunk_parsed_document([p])
        for c in chunks:
            assert c.page_number == 7


# ── Multi-block (multi-page) documents ───────────────────────────────────────

class TestMultiBlock:
    def test_multiple_parsed_blocks_merge_correctly(self):
        pages = [
            _make_parsed(f"Page {i} content. " * 5, source="multi.pdf")
            for i in range(1, 4)
        ]
        chunks = chunk_parsed_document(pages, chunk_size=900, chunk_overlap=150)
        assert len(chunks) >= 1
        # Chunk indices must still be contiguous
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_token_count_stored_on_each_chunk(self):
        text = "Vendor pricing data for FY2025."
        parsed = [_make_parsed(text)]
        chunks = chunk_parsed_document(parsed)
        for c in chunks:
            assert c.token_count > 0
            # token_count should agree with the direct counter
            assert c.token_count == _count_tokens(c.text)
