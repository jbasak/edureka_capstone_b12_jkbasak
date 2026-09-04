"""Extract searchable text while retaining page and sheet provenance."""

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


@dataclass(frozen=True)
class ParsedSection:
    filename: str
    text: str
    location: str


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx"}


def parse_document(filename: str, content: bytes) -> list[ParsedSection]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension or 'unknown'}'. Use: {allowed}")
    if not content:
        raise ValueError("Uploaded document is empty")

    if extension == ".pdf":
        reader = PdfReader(BytesIO(content))
        return [
            ParsedSection(filename, page.extract_text() or "", f"Page {number}")
            for number, page in enumerate(reader.pages, start=1)
        ]
    if extension == ".txt":
        return [ParsedSection(filename, content.decode("utf-8", errors="replace"), "Text")]
    if extension == ".csv":
        frame = pd.read_csv(StringIO(content.decode("utf-8-sig", errors="replace")))
        return [ParsedSection(filename, frame.to_csv(index=False), "Sheet 1")]

    workbook = pd.ExcelFile(BytesIO(content), engine="openpyxl")
    sections: list[ParsedSection] = []
    for sheet_name in workbook.sheet_names:
        frame = workbook.parse(sheet_name=sheet_name)
        sections.append(ParsedSection(filename, frame.to_csv(index=False), f"Sheet {sheet_name}"))
    return sections