# Specification:
## Procurement & Vendor Evaluation Assistant

### 1. Overview & ContextProblem Statement
Procurement and vendor management teams often spend hours or days manually scanning long, unstructured request-for-proposal (RFP) responses, complex pricing sheets, and master services agreements (MSAs) to ensure compliance and evaluate quotes.

#### Goals
* Automatically extract, normalize, and query compliance criteria from diverse vendor documents (PDF, TXT, CSV, XLSX).
* Leverage a multi-agent system to reason across requirements, cross-reference pricing models, and identify missing or non-compliant sections.
* Provide a grounded response backed by strict source citations to eliminate hallucinations.
* Package the application inside a reproducible Docker environment driven by a FastAPI backend.

#### Non-Goals
* This tool will not handle direct vendor communications or automated contract signing.
* It does not replace human legal review; it acts strictly as an analytical decision-support tool.

### 2. System Agents & Roles
The core engine utilizes an Agent Controller pattern to orchestrate four dedicated specialist agents:

#### 2.1 Requirement Extraction Agent: Parses RFP proposals, tables, and clauses to tag line-item criteria.** **Retrieval

#### 2.2 Agent: Queries the Vector DB with hybrid semantics to pull contextual evidence matching user queries.

#### 2.3 Comparison & Reasoning Agent: Analyzes multi-vendor structures, formats structured evaluation tables, and detects discrepancies.

#### 2.4 Validation Agent: Evaluates the final response against retrieved chunks to verify there are zero hallucinations, and enforces that all assertions contain strict source tracking.

### 3. Data Flow & Core Workflows

#### 3.1 Document Ingestion Flow

> [Parsing (PyMuPDF/pandas)] 
  ──> [Normalization & Chunking] ──> [Embeddings API] ──> [Qdrant Vector DB]
```

1. **Trigger:** User sends a file payload via the `/documents` endpoint.
2. **Parsing & Normalization:** 
   * **PDF/TXT:** Extracted line-by-line using `PyMuPDF` while maintaining structural headings.
   * **CSV/XLSX:** Parsed via `pandas`/`openpyxl` into unified row-by-row string chunks preserves row relations.
3. **Chunking & Vector Storage:** Chunks are token-counted, embedded via the OpenAI-compatible API, and inserted into a unified index in `Qdrant` tagged with metadata (`source_file`, `page_number`/`row_index`).

### 3.2 Multi-Agent Reasoning & Query Flow
1. **Trigger:** User posts a search question or comparative query to `/chat`.
2. **Task Planning:** The **Agent Controller** parses the query to determine if it is *Extractive* (single file) or *Comparative* (cross-vendor comparison).
3. **Evidence Retrieval:** The **Retrieval Agent** pulls the top-$K$ highly relevant context blocks from the vector database.
4. **Reasoning Engine:** The **Reasoning Agent** reads the raw text, aligns requirements against the text, and designs a Markdown synthesis or matrix comparison.
5. **Validation Loop:** The **Validation Agent** scans the output text. If any assertion lacks a corresponding reference matching the vector database block metadata, it triggers a rewrite before sending it to the user.

---

## 4. Functional Requirements

### 4.1 Document Handling & Chunking
* [ ] **Multi-Format Support:** Correctly parse text from raw `.txt`, `.pdf` layout files, `.csv` sheets, and multi-tab Excel files (`.xlsx`).
* [ ] **Overlapping Sliders:** Chunk sizes should target ~500 tokens with a 10% overlap to preserve bounding sentence contexts.

### 4.2 Agent Execution & Evaluation Guardrails
* [ ] **Requirement Checklist Extractor:** Given an Excel sheet of compliance requirements, map whether Vendor A, B, or C satisfies the condition (e.g., "Pass", "Fail", or "Partially Met").
* [ ] **Hallucination Minimization:** The system must return an explicit *"Information not found in context"* string if the Vector DB does not provide an explicit answer.
* [ ] **Strict Citation Injector:** Every analytical text output must include bracketed citations linked to the metadata payload (e.g., `[VendorA_Proposal.pdf, Page 12]`).

---

## 5. Technical Architecture & Stack
* **Language/Framework:** Python 3.11 + FastAPI (Pydantic schemas for request/response validation).
* **Vector DB:** Qdrant (Persistent localized volume or cloud instance).
* **AI Model Routing:** OpenAI-compatible API configuration (configurable via local `.env` variables).
* **Infrastructure:** Minimalistic `docker-compose.yml` defining the FastAPI worker service and the Qdrant instance sidecar.

---

## 6. Verification & Acceptance Criteria
These operational criteria must pass successfully during localized testing:

* [ ] **Scenario 1 (Ingestion):** Given an upload payload of `acme_proposal.pdf` and `pricing.csv`, when sent to `/documents`, then the system successfully commits vectors to Qdrant and yields a `201 Created` status.
* [ ] **Scenario 2 (Comparative Search):** Given a multi-vendor dataset, when queried with *"Compare annual pricing"*, then the system yields a markdown table outlining costs explicitly matching the input spreadsheet values.
* [ ] **Scenario 3 (Citation Validation):** When a response is generated, then every factual claim contains at least one target file source citation.

***

### Next Steps with Claude Code:
You can now create a file called `specs/procurement_agent.md` with this text and pass it to Claude Code. 

If you are ready to build, let me know:
* Would you like me to generate the foundational **`docker-compose.yml`** and **`.env.example`** to kick off the container structure?
* Do you want the baseline **FastAPI scaffolding codebase** (`main.py` router templates) for processing the chunk metadata?
