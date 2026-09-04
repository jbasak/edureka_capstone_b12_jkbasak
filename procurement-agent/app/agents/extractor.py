"""
app/agents/extractor.py

Requirement Extraction Agent
-----------------------------
Specialises in Excel/CSV compliance checklists.

During ingestion it stores requirement rows the same way as all other documents
(via the shared ingestion pipeline).  At query time it can be asked to produce a
structured compliance matrix: for each extracted requirement, it checks whether
each vendor satisfies it ("Pass", "Fail", "Partially Met", or "Not Found").

The agent uses the RetrievalAgent to pull evidence for each requirement and the
ReasoningAgent to produce the verdict — keeping it grounded and cited.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Requirement:
    """A single compliance requirement extracted from an Excel/CSV checklist."""

    requirement_id: str      # e.g. "REQ-001" or row number
    description: str         # full requirement text
    category: Optional[str]  # optional grouping column value
    source_file: str
    row_index: int


@dataclass
class ComplianceVerdict:
    """Verdict for one requirement × one vendor."""

    requirement_id: str
    vendor_name: str
    status: str              # "Pass" | "Fail" | "Partially Met" | "Not Found"
    evidence: str            # brief text excerpt supporting the verdict
    citation: str            # e.g. "[AWS_Proposal.pdf, Page 4]"


# ── Agent ─────────────────────────────────────────────────────────────────────

class RequirementExtractionAgent:
    """
    Extracts structured requirements from Excel/CSV files and evaluates
    vendor compliance using retrieval-augmented reasoning.
    """

    # Keywords that typically mark a requirements column
    _REQ_COL_HINTS = [
        "requirement", "criteria", "question", "description",
        "specification", "item", "req", "feature",
    ]
    # Keywords that typically mark an ID column
    _ID_COL_HINTS = ["id", "no", "number", "#", "ref", "item"]

    def extract_requirements(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> list[Requirement]:
        """
        Parse an Excel (.xlsx) or CSV file and extract requirement rows.

        Heuristically identifies the requirement description column and an
        optional ID column.  All other columns are treated as category metadata.

        Returns
        -------
        List of Requirement objects (one per non-empty row).
        """
        import pandas as pd

        ext = filename.lower().rsplit(".", 1)[-1]

        if ext == "csv":
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
            except Exception:
                df = pd.read_csv(
                    io.BytesIO(file_bytes), dtype=str, keep_default_na=False, encoding="latin-1"
                )
        elif ext == "xlsx":
            df = pd.read_excel(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
        else:
            raise ValueError(f"RequirementExtractionAgent supports .csv and .xlsx only, got .{ext}")

        # Normalise column names for heuristic matching
        col_lower = {col: col.lower().strip() for col in df.columns}

        req_col = self._find_column(col_lower, self._REQ_COL_HINTS)
        id_col = self._find_column(col_lower, self._ID_COL_HINTS)
        # Pick first column that isn't req or id for category
        cat_col = next(
            (c for c in df.columns if c not in {req_col, id_col}), None
        )

        if req_col is None:
            # Fall back: use the widest text column
            req_col = max(df.columns, key=lambda c: df[c].str.len().max())
            logger.warning(
                "No obvious requirement column found in '%s'; using '%s'.",
                filename,
                req_col,
            )

        requirements: list[Requirement] = []
        for idx, row in df.iterrows():
            desc = str(row.get(req_col, "")).strip()
            if not desc or desc.lower() in ("nan", "none", ""):
                continue
            req_id = (
                str(row.get(id_col, "")).strip() if id_col else f"REQ-{int(idx)+1:04d}"
            )
            if not req_id or req_id.lower() in ("nan", "none", ""):
                req_id = f"REQ-{int(idx)+1:04d}"

            category = str(row.get(cat_col, "")).strip() if cat_col else None

            requirements.append(
                Requirement(
                    requirement_id=req_id,
                    description=desc,
                    category=category,
                    source_file=filename,
                    row_index=int(idx),
                )
            )

        logger.info(
            "RequirementExtractionAgent: extracted %d requirements from '%s'.",
            len(requirements),
            filename,
        )
        return requirements

    def build_compliance_matrix(
        self,
        requirements: list[Requirement],
        vendor_names: list[str],
        retrieval_agent,
        reasoning_agent,
        top_k: int = 3,
    ) -> list[ComplianceVerdict]:
        """
        For each (requirement, vendor) pair, retrieve evidence and produce a
        Pass / Fail / Partially Met / Not Found verdict.

        Parameters
        ----------
        requirements     : extracted requirement list
        vendor_names     : vendors to evaluate (e.g. ["AWS", "Google", "Oracle"])
        retrieval_agent  : RetrievalAgent instance
        reasoning_agent  : ReasoningAgent instance
        top_k            : chunks to retrieve per (requirement, vendor) pair

        Returns
        -------
        List of ComplianceVerdict objects.
        """
        verdicts: list[ComplianceVerdict] = []

        for req in requirements:
            for vendor in vendor_names:
                query = f"Does {vendor} satisfy the following requirement: {req.description}"
                chunks = retrieval_agent.retrieve(
                    question=query,
                    vendor_filter=[vendor],
                    top_k=top_k,
                )

                if not chunks:
                    verdicts.append(
                        ComplianceVerdict(
                            requirement_id=req.requirement_id,
                            vendor_name=vendor,
                            status="Not Found",
                            evidence="No relevant context found in indexed documents.",
                            citation="",
                        )
                    )
                    continue

                # Ask the reasoning agent for a single-sentence verdict
                verdict_question = (
                    f"Based only on the provided context, does {vendor} satisfy "
                    f"this requirement: '{req.description}'? "
                    f"Answer with one of: Pass, Fail, Partially Met, or Not Found. "
                    f"Then provide a one-sentence justification with a citation."
                )
                draft = reasoning_agent.reason(
                    question=verdict_question,
                    chunks=chunks,
                    mode="extractive",
                )

                # Parse status keyword from the response
                status = "Not Found"
                for keyword in ("Pass", "Fail", "Partially Met"):
                    if keyword.lower() in draft.lower():
                        status = keyword
                        break

                citation = chunks[0].citation_label() if chunks else ""
                verdicts.append(
                    ComplianceVerdict(
                        requirement_id=req.requirement_id,
                        vendor_name=vendor,
                        status=status,
                        evidence=draft[:300],
                        citation=citation,
                    )
                )

        logger.info(
            "RequirementExtractionAgent: produced %d compliance verdicts.",
            len(verdicts),
        )
        return verdicts

    def format_compliance_table(
        self,
        requirements: list[Requirement],
        verdicts: list[ComplianceVerdict],
        vendor_names: list[str],
    ) -> str:
        """
        Render the compliance matrix as a Markdown table.

        Example output
        --------------
        | Req ID | Description | AWS | Google | Oracle |
        |--------|-------------|-----|--------|--------|
        | REQ-001| Uptime 99.9%| Pass| Fail   | Pass   |
        """
        verdict_map: dict[tuple[str, str], ComplianceVerdict] = {
            (v.requirement_id, v.vendor_name): v for v in verdicts
        }

        header = "| Req ID | Description | " + " | ".join(vendor_names) + " |"
        sep = "|--------|-------------|" + "|".join(["------"] * len(vendor_names)) + "|"
        rows = [header, sep]

        for req in requirements:
            desc = req.description[:60] + ("…" if len(req.description) > 60 else "")
            cells = []
            for vendor in vendor_names:
                v = verdict_map.get((req.requirement_id, vendor))
                cells.append(v.status if v else "Not Found")
            rows.append(
                f"| {req.requirement_id} | {desc} | " + " | ".join(cells) + " |"
            )

        return "\n".join(rows)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_column(
        self,
        col_lower: dict[str, str],
        hints: list[str],
    ) -> Optional[str]:
        """Return the first column whose lowercased name contains any hint word."""
        for original, lower in col_lower.items():
            for hint in hints:
                if hint in lower:
                    return original
        return None
