"""
app/ingestion/parser.py

Multi-format document parser.  Returns a flat list of ParsedChunk objects,
each carrying raw text and provenance metadata before token-level splitting.

Supported formats
-----------------
  .pdf   – PyMuPDF (fitz), page-by-page with heading preservation
  .txt   – built-in open(), treated as a single continuous block
  .csv   – pandas, each row converted to "col: val | col: val …" string
  .xlsx  – pandas + openpyxl, all sheets iterated
  .docx  – python-docx, paragraphs + table cells in order
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported MIME / extension map
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".docx"}


@dataclass
class ParsedChunk:
    """Raw text block produced by the parser (before token splitting)."""

    text: str
    source_file: str
    vendor_name: Optional[str] = None
    doc_category: Optional[str] = None
    page_number: Optional[int] = None   # PDF / DOCX pages (1-based)
    row_index: Optional[int] = None     # CSV / XLSX rows (0-based)
    sheet_name: Optional[str] = None    # XLSX sheet label
    extra: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.text.strip()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_document(
    file_bytes: bytes,
    filename: str,
    vendor_name: Optional[str] = None,
    doc_category: Optional[str] = None,
) -> list[ParsedChunk]:
    """
    Dispatch to the correct parser based on file extension.

    Parameters
    ----------
    file_bytes   : raw bytes of the uploaded file
    filename     : original filename (used for extension detection + metadata)
    vendor_name  : optional vendor tag propagated to every chunk
    doc_category : optional category tag propagated to every chunk

    Returns
    -------
    List of ParsedChunk objects; empty chunks are filtered out.

    Raises
    ------
    ValueError   : unsupported file extension
    RuntimeError : parsing failure with details
    """
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    logger.info("Parsing '%s' as %s (vendor=%s, category=%s)",
                filename, ext, vendor_name, doc_category)

    common = dict(
        source_file=filename,
        vendor_name=vendor_name,
        doc_category=doc_category,
    )

    try:
        if ext == ".pdf":
            chunks = _parse_pdf(file_bytes, **common)
        elif ext == ".txt":
            chunks = _parse_txt(file_bytes, **common)
        elif ext == ".csv":
            chunks = _parse_csv(file_bytes, **common)
        elif ext == ".xlsx":
            chunks = _parse_xlsx(file_bytes, **common)
        elif ext == ".docx":
            chunks = _parse_docx(file_bytes, **common)
        else:
            chunks = []
    except Exception as exc:
        raise RuntimeError(f"Failed to parse '{filename}': {exc}") from exc

    # Filter blank chunks
    result = [c for c in chunks if not c.is_empty()]
    logger.info("Parsed '%s' → %d non-empty raw blocks", filename, len(result))
    return result


# ── PDF ───────────────────────────────────────────────────────────────────────

def _parse_pdf(
    file_bytes: bytes,
    source_file: str,
    vendor_name: Optional[str],
    doc_category: Optional[str],
) -> list[ParsedChunk]:
    """Extract text page-by-page using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for PDF parsing: pip install pymupdf") from exc

    chunks: list[ParsedChunk] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # preserve layout order
            if text.strip():
                chunks.append(
                    ParsedChunk(
                        text=text,
                        source_file=source_file,
                        vendor_name=vendor_name,
                        doc_category=doc_category,
                        page_number=page_num,
                    )
                )
    return chunks


# ── TXT ───────────────────────────────────────────────────────────────────────

def _parse_txt(
    file_bytes: bytes,
    source_file: str,
    vendor_name: Optional[str],
    doc_category: Optional[str],
) -> list[ParsedChunk]:
    """Decode UTF-8 (with latin-1 fallback) and return as a single block."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    return [
        ParsedChunk(
            text=text,
            source_file=source_file,
            vendor_name=vendor_name,
            doc_category=doc_category,
        )
    ]


# ── CSV ───────────────────────────────────────────────────────────────────────

def _parse_csv(
    file_bytes: bytes,
    source_file: str,
    vendor_name: Optional[str],
    doc_category: Optional[str],
) -> list[ParsedChunk]:
    """
    Convert each CSV row to a natural-language-style string:
      "ColumnA: valueA | ColumnB: valueB | …"
    Preserves header-to-value relationships for retrieval.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for CSV parsing.") from exc

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
    except Exception:
        # Try latin-1 if UTF-8 fails
        df = pd.read_csv(
            io.BytesIO(file_bytes), dtype=str, keep_default_na=False, encoding="latin-1"
        )

    chunks: list[ParsedChunk] = []
    for row_idx, row in df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
        text = " | ".join(parts)
        if text.strip():
            chunks.append(
                ParsedChunk(
                    text=text,
                    source_file=source_file,
                    vendor_name=vendor_name,
                    doc_category=doc_category,
                    row_index=int(row_idx),
                )
            )
    return chunks


# ── XLSX ──────────────────────────────────────────────────────────────────────

def _parse_xlsx(
    file_bytes: bytes,
    source_file: str,
    vendor_name: Optional[str],
    doc_category: Optional[str],
) -> list[ParsedChunk]:
    """
    Iterate every sheet in the workbook.
    Each row → "SheetName | col: val | col: val …"
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas + openpyxl are required for XLSX parsing.") from exc

    chunks: list[ParsedChunk] = []
    xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name, dtype=str, keep_default_na=False)
        for row_idx, row in df.iterrows():
            parts = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
            if not parts:
                continue
            text = f"[Sheet: {sheet_name}] " + " | ".join(parts)
            chunks.append(
                ParsedChunk(
                    text=text,
                    source_file=source_file,
                    vendor_name=vendor_name,
                    doc_category=doc_category,
                    row_index=int(row_idx),
                    sheet_name=sheet_name,
                )
            )
    return chunks


# ── DOCX ──────────────────────────────────────────────────────────────────────

def _parse_docx(
    file_bytes: bytes,
    source_file: str,
    vendor_name: Optional[str],
    doc_category: Optional[str],
) -> list[ParsedChunk]:
    """
    Extract paragraphs and table cell text from a .docx file.
    Groups content by approximate page boundaries (every 40 paragraphs)
    since python-docx does not expose page numbers directly.
    """
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("python-docx is required for DOCX parsing.") from exc

    doc = Document(io.BytesIO(file_bytes))
    chunks: list[ParsedChunk] = []
    page_size = 40  # paragraphs per synthetic "page"
    buffer: list[str] = []
    page_num = 1

    def flush_buffer(buf: list[str], pn: int) -> None:
        text = "\n".join(buf).strip()
        if text:
            chunks.append(
                ParsedChunk(
                    text=text,
                    source_file=source_file,
                    vendor_name=vendor_name,
                    doc_category=doc_category,
                    page_number=pn,
                )
            )

    # Paragraphs
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            buffer.append(para.text.strip())
        if len(buffer) >= page_size:
            flush_buffer(buffer, page_num)
            buffer = []
            page_num += 1

    # Tables (append after paragraphs)
    for table in doc.tables:
        for row in table.rows:
            cell_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cell_texts:
                buffer.append(" | ".join(cell_texts))
            if len(buffer) >= page_size:
                flush_buffer(buffer, page_num)
                buffer = []
                page_num += 1

    # Flush remaining
    flush_buffer(buffer, page_num)
    return chunks
