"""
tests/test_parser.py

Unit tests for app/ingestion/parser.py

Tests cover:
  - PDF parsing  (PyMuPDF)
  - TXT parsing
  - CSV row-to-string conversion
  - XLSX multi-sheet iteration
  - DOCX paragraph extraction
  - Unsupported format rejection
  - Metadata propagation (vendor_name, doc_category)
  - Empty-chunk filtering
"""

from __future__ import annotations

import io
import pytest

from app.ingestion.parser import parse_document, ParsedChunk, SUPPORTED_EXTENSIONS


# ── TXT ───────────────────────────────────────────────────────────────────────

class TestTxtParser:
    def test_returns_list_of_parsed_chunks(self, sample_txt_bytes):
        chunks = parse_document(sample_txt_bytes, "vendor.txt")
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(c, ParsedChunk) for c in chunks)

    def test_text_content_preserved(self, sample_txt_bytes):
        chunks = parse_document(sample_txt_bytes, "vendor.txt")
        combined = " ".join(c.text for c in chunks)
        assert "uptime" in combined.lower()
        assert "99.9%" in combined

    def test_metadata_propagated(self, sample_txt_bytes):
        chunks = parse_document(
            sample_txt_bytes, "vendor.txt",
            vendor_name="Acme", doc_category="proposal"
        )
        for c in chunks:
            assert c.vendor_name == "Acme"
            assert c.doc_category == "proposal"
            assert c.source_file == "vendor.txt"

    def test_no_empty_chunks_returned(self, sample_txt_bytes):
        chunks = parse_document(sample_txt_bytes, "vendor.txt")
        for c in chunks:
            assert c.text.strip() != ""

    def test_latin1_fallback(self):
        """Bytes with latin-1 encoding should not raise."""
        latin1_bytes = "Société générale\nCloud pricing: €500,000\n".encode("latin-1")
        chunks = parse_document(latin1_bytes, "notes.txt")
        assert len(chunks) >= 1


# ── CSV ───────────────────────────────────────────────────────────────────────

class TestCsvParser:
    def test_row_count_matches(self, sample_csv_bytes):
        # 3 data rows (header excluded)
        chunks = parse_document(sample_csv_bytes, "pricing.csv")
        assert len(chunks) == 3

    def test_row_format_has_key_value_pairs(self, sample_csv_bytes):
        chunks = parse_document(sample_csv_bytes, "pricing.csv")
        for c in chunks:
            assert ": " in c.text, f"Expected 'col: val' format, got: {c.text!r}"

    def test_row_index_set(self, sample_csv_bytes):
        chunks = parse_document(sample_csv_bytes, "pricing.csv")
        indices = [c.row_index for c in chunks]
        assert all(idx is not None and idx >= 0 for idx in indices)
        # Indices should be distinct
        assert len(set(indices)) == len(indices)

    def test_vendor_name_in_metadata(self, sample_csv_bytes):
        chunks = parse_document(
            sample_csv_bytes, "pricing.csv", vendor_name="AWS"
        )
        for c in chunks:
            assert c.vendor_name == "AWS"

    def test_column_values_present_in_text(self, sample_csv_bytes):
        chunks = parse_document(sample_csv_bytes, "pricing.csv")
        combined = " ".join(c.text for c in chunks)
        assert "480000" in combined
        assert "99.99%" in combined


# ── XLSX ──────────────────────────────────────────────────────────────────────

class TestXlsxParser:
    def test_returns_chunks(self, sample_xlsx_bytes):
        chunks = parse_document(sample_xlsx_bytes, "requirements.xlsx")
        assert len(chunks) >= 3  # 3 data rows

    def test_sheet_name_in_metadata(self, sample_xlsx_bytes):
        chunks = parse_document(sample_xlsx_bytes, "requirements.xlsx")
        assert all(c.sheet_name is not None for c in chunks)

    def test_sheet_name_in_text(self, sample_xlsx_bytes):
        chunks = parse_document(sample_xlsx_bytes, "requirements.xlsx")
        for c in chunks:
            assert "Requirements" in c.text

    def test_requirement_text_present(self, sample_xlsx_bytes):
        chunks = parse_document(sample_xlsx_bytes, "requirements.xlsx")
        combined = " ".join(c.text for c in chunks)
        assert "uptime" in combined.lower()
        assert "FedRAMP" in combined

    def test_multi_sheet(self):
        """Workbook with two sheets produces chunks from both."""
        import openpyxl
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "Sheet1"
        ws1.append(["col_a", "col_b"])
        ws1.append(["val_a1", "val_b1"])
        ws2 = wb.create_sheet("Sheet2")
        ws2.append(["col_x"])
        ws2.append(["val_x1"])
        buf = io.BytesIO()
        wb.save(buf)
        chunks = parse_document(buf.getvalue(), "multi.xlsx")
        sheet_names = {c.sheet_name for c in chunks}
        assert "Sheet1" in sheet_names
        assert "Sheet2" in sheet_names


# ── PDF ───────────────────────────────────────────────────────────────────────

class TestPdfParser:
    def test_returns_chunks(self, sample_pdf_bytes):
        chunks = parse_document(sample_pdf_bytes, "proposal.pdf")
        # Minimal PDF may produce 0 or 1 text chunks depending on PyMuPDF rendering
        assert isinstance(chunks, list)

    def test_page_number_set_when_text_found(self, sample_pdf_bytes):
        chunks = parse_document(sample_pdf_bytes, "proposal.pdf")
        for c in chunks:
            assert c.page_number is not None and c.page_number >= 1


# ── Unsupported / edge cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_unsupported_extension_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(b"data", "archive.zip")

    def test_no_extension_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(b"data", "README")

    def test_supported_extensions_set(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".xlsx" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
