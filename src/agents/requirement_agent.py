"""Compliance spreadsheet normalization and vendor status evaluation."""

from io import BytesIO
import re

import pandas as pd


STATUSES = ("Pass", "Fail", "Partially Met")


def evaluate_requirements(content: bytes, filename: str = "requirements.xlsx") -> list[dict[str, str]]:
    if not filename.lower().endswith(".xlsx"):
        raise ValueError("Compliance evaluation requires an .xlsx workbook")
    workbook = pd.ExcelFile(BytesIO(content), engine="openpyxl")
    rows: list[dict[str, str]] = []
    for sheet in workbook.sheet_names:
        frame = workbook.parse(sheet).fillna("")
        for _, row in frame.iterrows():
            values = {str(key).strip(): str(value).strip() for key, value in row.items()}
            requirement = values.get("Requirement") or values.get("Criteria") or next(iter(values.values()), "")
            result = {"requirement": requirement, "source": f"{filename}, Sheet {sheet}"}
            for vendor in ("Vendor A", "Vendor B", "Vendor C"):
                value = next((text for key, text in values.items() if vendor.lower() in key.lower()), "")
                result[vendor] = _status(value)
            rows.append(result)
    if not rows:
        raise ValueError("Compliance workbook contains no data rows")
    return rows


def _status(value: str) -> str:
    normalized = re.sub(r"[^a-z ]", "", value.lower()).strip()
    if normalized in {"yes", "pass", "compliant", "met", "true"}:
        return "Pass"
    if normalized in {"no", "fail", "non compliant", "not met", "false"}:
        return "Fail"
    return "Partially Met"