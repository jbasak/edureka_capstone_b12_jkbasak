# PLAN.md — Procurement & Vendor Evaluation Assistant
## Technical Implementation Plan

> **Project:** Edureka Capstone B12 — AI Procurement Agent  
> **Stack:** Python 3.11 · FastAPI · Qdrant · Groq API (LLM) · HuggingFace Sentence Transformers (embeddings) · Docker Compose  
> **Author:** Data Engineer  
> **Date:** 2026-09-03

---

## 1. Project Structure

```
procurement-agent/
├── .env                          # API keys and config (already present)
├── .env.example                  # Template (no secrets)
├── docker-compose.yml            # FastAPI + Qdrant services
├── Dockerfile                    # FastAPI app image
├── requirements.txt              # Python dependencies (pinned)
├── PLAN.md                       # This document
│
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   ├── config.py                 # Pydantic Settings (reads .env)
│   ├── dependencies.py           # Shared FastAPI dependency injectors
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── documents.py          # POST /documents router
│   │   └── chat.py               # POST /chat router
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── controller.py         # Agent Controller (orchestration logic)
│   │   ├── extractor.py          # Requirement Extraction Agent
│   │   ├── retrieval.py          # Retrieval Agent (Qdrant queries)
│   │   ├── reasoning.py          # Comparison & Reasoning Agent (Groq LLM)
│   │   └── validator.py          # Validation Agent (citation checker)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py             # Multi-format document parser
│   │   ├── chunker.py            # Token-aware chunker with overlap
│   │   └── embedder.py           # HuggingFace embedding wrapper
│   │
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── qdrant_client.py      # Qdrant collection management + upsert/search
│   │
│   └── schemas/
│       ├── __init__.py
│       ├── document.py           # Pydantic models for ingestion
│       └── chat.py               # Pydantic models for chat
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Shared fixtures, mock Qdrant/Groq
│   ├── test_parser.py            # Unit tests: parser
│   ├── test_chunker.py           # Unit tests: chunker
│   ├── test_embedder.py          # Unit tests: embedder
│   ├── test_ingestion_api.py     # Integration: POST /documents
│   ├── test_chat_api.py          # Integration: POST /chat
│   └── test_agents.py            # Unit tests: each agent
│
├── specs/
│   ├── procurement-agent.md
│   └── capstone_guidelines.md
│
└── testdata/                     # Sample documents for manual + automated testing
    ├── rfp_award_letter.pdf
    ├── rfp_documents/
    │   ├── Cloud RFP Bidder Questions.xlsx
    │   ├── Cloud Services RFP 250000000414 Questions Round 2.xlsx
    │   ├── Recent Purchase History.csv
    │   ├── RFP 250000000414 - Cloud Services 12.19.24 version.docx
    │   └── RFP 250000000414 - Cloud Services.docx
    ├── rfp_response_AWS/
    │   ├── AWS Response to Michigan DTMB Cloud Services RFP Schedule B Pricing.pdf
    │   └── AWS Response to Michigan DTMB Cloud Services RFP_Public.pdf
    ├── rfp_response_Google/
    │   ├── Appendix 1 -Schedule A - Technical Req 6.0 Answer.pdf
    │   ├── Google Cover Letter.pdf
    │   ├── Google RFP 250000000414 - Cloud Services 12.19.24 version (1)_Public.pdf
    │   └── SecOps Services Schedule Template - MICH RFP 250000000414 - Cloud Services.pdf
    └── rfp_response_Oracle/
        ├── Oracle General Terms and Conditions ... .pdf
        ├── RFP 250000000414 Cloud Services ... Oracle Redacted 1 of 3.pdf
        ├── Schedule B Pricing-Oracle.pdf
        └── (+ other Oracle docs)
```

---

## 2. Environment Configuration

### 2.1 `.env` Variables (already configured)

| Variable | Value | Purpose |
|---|---|---|
| `GROQ_API_KEY` | `gsk_...` | Groq cloud LLM API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model for reasoning |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace local embedding model |
| `CHUNK_SIZE` | `900` | Tokens per chunk (spec: ~500, increased for richer context) |
| `CHUNK_OVERLAP` | `150` | Overlap tokens (~17%, meets spec 10% minimum) |
| `TOP_K` | `6` | Top retrieved chunks per query |
| `MAX_UPLOAD_MB` | `25` | Max file upload size |
| `LOG_LEVEL` | `INFO` | Application log level |

### 2.2 Additional `.env` Variables to Add

```env
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=procurement_docs
VECTOR_DIMENSION=384          # all-MiniLM-L6-v2 output dimension
GROQ_BASE_URL=https://api.groq.com/openai/v1
APP_PORT=8000
```

### 2.3 `.env.example` (committed to repo, no secrets)

Identical to `.env` with all secret values replaced by placeholders.

---

## 3. API Schema Definitions

### 3.1 `POST /documents` — Document Ingestion

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` | Yes | The document to ingest (PDF, TXT, CSV, XLSX, DOCX) |
| `vendor_name` | `string` | No | Tag this document with a vendor label (e.g., `AWS`, `Google`, `Oracle`) |
| `doc_category` | `string` | No | Category tag (e.g., `proposal`, `pricing`, `requirements`, `contract`) |

**Success Response — `201 Created`**

```json
{
  "status": "success",
  "filename": "AWS Response to Michigan DTMB Cloud Services RFP_Public.pdf",
  "vendor_name": "AWS",
  "doc_category": "proposal",
  "chunks_stored": 47,
  "collection": "procurement_docs"
}
```

**Error Response — `422 Unprocessable Entity`**

```json
{
  "detail": "Unsupported file type: .docx is not in [pdf, txt, csv, xlsx]"
}
```

**Error Response — `413 Request Entity Too Large`**

```json
{
  "detail": "File size 30MB exceeds maximum allowed 25MB"
}
```

**Pydantic Schema (`app/schemas/document.py`)**

```python
class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    vendor_name: Optional[str] = None
    doc_category: Optional[str] = None
    chunks_stored: int
    collection: str

class ChunkMetadata(BaseModel):
    source_file: str
    vendor_name: Optional[str]
    doc_category: Optional[str]
    page_number: Optional[int]      # for PDF/DOCX
    row_index: Optional[int]        # for CSV/XLSX
    chunk_index: int
    total_chunks: int
```

---

### 3.2 `POST /chat` — Agent Query

**Request:** `application/json`

```json
{
  "question": "Compare annual pricing across AWS, Google, and Oracle",
  "vendor_filter": ["AWS", "Google", "Oracle"],
  "mode": "comparative"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | `string` | Yes | Natural-language query |
| `vendor_filter` | `list[string]` | No | Limit retrieval to specific vendors |
| `mode` | `enum` | No | `extractive` \| `comparative` (auto-detected if omitted) |

**Success Response — `200 OK`**

```json
{
  "answer": "## Pricing Comparison\n\n| Vendor | Annual Cost | Notes |\n|...|...|...|\n\n**Sources:**\n- [AWS Schedule B Pricing.pdf, Page 3]\n- [Oracle Schedule B Pricing.pdf, Page 5]\n- [Google RFP Response, Page 8]",
  "mode": "comparative",
  "sources": [
    {
      "source_file": "AWS Response to Michigan DTMB Cloud Services RFP Schedule B Pricing.pdf",
      "vendor_name": "AWS",
      "page_number": 3,
      "chunk_index": 12,
      "score": 0.91
    }
  ],
  "validation_passed": true,
  "citations_count": 3
}
```

**Fallback Response (no context found)**

```json
{
  "answer": "Information not found in context.",
  "mode": "extractive",
  "sources": [],
  "validation_passed": true,
  "citations_count": 0
}
```

**Pydantic Schema (`app/schemas/chat.py`)**

```python
class ChatRequest(BaseModel):
    question: str
    vendor_filter: Optional[List[str]] = None
    mode: Optional[Literal["extractive", "comparative"]] = None

class SourceReference(BaseModel):
    source_file: str
    vendor_name: Optional[str]
    page_number: Optional[int]
    row_index: Optional[int]
    chunk_index: int
    score: float

class ChatResponse(BaseModel):
    answer: str
    mode: str
    sources: List[SourceReference]
    validation_passed: bool
    citations_count: int
```

---

### 3.3 `GET /health` — Health Check

**Response — `200 OK`**

```json
{
  "status": "healthy",
  "qdrant": "connected",
  "collection": "procurement_docs",
  "vector_count": 312,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "llama-3.3-70b-versatile"
}
```

---

## 4. Module-Level Implementation Details

### 4.1 `app/config.py` — Settings

Use `pydantic_settings.BaseSettings` to load all `.env` variables with type validation. Single instantiated `Settings` object imported across the app to avoid repeated env reads.

---

### 4.2 `app/ingestion/parser.py` — Multi-Format Parser

| File Type | Library | Behavior |
|---|---|---|
| `.pdf` | `PyMuPDF` (fitz) | Extract page-by-page text; preserve heading structure; attach `page_number` metadata |
| `.txt` | built-in `open()` | Read raw text; treat entire file as single block before chunking |
| `.csv` | `pandas` | Read all rows; convert each row to a text string `"col1: val1 \| col2: val2 ..."`; attach `row_index` |
| `.xlsx` | `pandas` + `openpyxl` | Iterate all sheets; same row-to-string conversion per sheet; include sheet name in metadata |
| `.docx` | `python-docx` | Extract paragraphs and table cells; preserve paragraph order |

**Output:** `list[ParsedChunk]` where each chunk is a raw text block with source metadata before token-splitting.

---

### 4.3 `app/ingestion/chunker.py` — Token-Aware Chunker

- Uses `tiktoken` (cl100k_base encoding) to count tokens.
- Splits text at sentence boundaries when possible to avoid mid-sentence cuts.
- `CHUNK_SIZE=900` tokens with `CHUNK_OVERLAP=150` tokens (~17%).
- Returns `list[TextChunk]` with `chunk_index` and inherited parent metadata.

---

### 4.4 `app/ingestion/embedder.py` — HuggingFace Embedder

- Loads `sentence-transformers/all-MiniLM-L6-v2` locally via the `sentence-transformers` library.
- Produces 384-dimensional dense vectors.
- Batches embedding calls in groups of 32 for memory efficiency.
- Model cached to disk to avoid re-downloading on container restart (mounted Docker volume).

---

### 4.5 `app/vector_store/qdrant_client.py` — Qdrant Interface

- Creates collection `procurement_docs` on startup if it does not exist, with cosine distance metric.
- `upsert_chunks(chunks, vectors, metadata)` — bulk inserts with UUID point IDs derived from `sha256(source_file + chunk_index)` for idempotency.
- `search(query_vector, top_k, vendor_filter)` — cosine similarity search with optional payload filter on `vendor_name`.
- Point payload mirrors `ChunkMetadata` schema.

---

### 4.6 `app/agents/controller.py` — Agent Controller

Orchestration logic:

```
1. Receive ChatRequest
2. Classify query mode: "extractive" vs "comparative"
   - "comparative" if question contains comparison keywords (compare, vs, difference, all vendors, etc.)
   - Otherwise "extractive"
3. Call Retrieval Agent → list[RetrievedChunk]
4. If len(chunks) == 0: return "Information not found in context."
5. Call Reasoning Agent with chunks + question + mode → draft_answer
6. Call Validation Agent with draft_answer + chunks → validated_answer + citations_ok
7. Return ChatResponse
```

---

### 4.7 `app/agents/extractor.py` — Requirement Extraction Agent

- Accepts an Excel file (e.g., `Cloud RFP Bidder Questions.xlsx`) as input.
- Extracts each row as a compliance requirement string.
- Stores requirement text into Qdrant tagged with `doc_category="requirements"`.
- Used during ingestion of requirements spreadsheets; also invoked to generate a compliance checklist at query time.

---

### 4.8 `app/agents/retrieval.py` — Retrieval Agent

- Embeds the user question using the same HuggingFace model.
- Calls `qdrant_client.search()` with `top_k=TOP_K` (default 6).
- Applies `vendor_filter` from request if present.
- Returns ranked `list[RetrievedChunk]` with scores and full metadata.

---

### 4.9 `app/agents/reasoning.py` — Reasoning Agent

- Constructs a system prompt instructing the LLM to:
  - Answer strictly from the provided context.
  - Return `"Information not found in context."` if evidence is absent.
  - Include bracketed citations `[filename, Page X]` for every factual claim.
  - Format comparative answers as Markdown tables.
- Calls Groq API at `https://api.groq.com/openai/v1/chat/completions` using `openai` SDK with `base_url` override.
- Model: `llama-3.3-70b-versatile` (from `.env`).

**System Prompt Template:**

```
You are a procurement analysis assistant. Answer the user's question ONLY using 
the provided context chunks. Each answer must include citations in the format 
[source_file, Page N] or [source_file, Row N]. If the context does not contain 
sufficient information, respond with exactly: "Information not found in context."
Do not infer or fabricate information.
```

---

### 4.10 `app/agents/validator.py` — Validation Agent

- Parses the LLM response text for bracketed citation patterns: `\[.+?,\s*(Page|Row)\s*\d+\]`.
- Cross-references each citation against the `source_file` values in the retrieved chunks.
- If any factual sentence (non-header, non-list-item line longer than 20 chars) has no associated citation: flags `validation_passed=False` and requests a rewrite (single retry).
- Returns `(validated_text, citations_count, validation_passed)`.

---

## 5. Docker Compose Architecture

### 5.1 Services

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.9.2
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]

  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [qdrant]
    volumes:
      - ./testdata:/app/testdata      # sample data accessible inside container
      - hf_cache:/root/.cache/huggingface  # persist embedding model cache
```

### 5.2 Dockerfile

- Base: `python:3.11-slim`
- Copy `requirements.txt` first (layer cache optimization), then `app/`.
- `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`

---

## 6. `requirements.txt` (pinned)

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
pydantic-settings==2.2.1
python-multipart==0.0.9
groq==0.8.0
openai==1.30.1
sentence-transformers==2.7.0
torch==2.3.0+cpu         # CPU-only to keep image small
qdrant-client==1.9.1
pymupdf==1.24.3
pandas==2.2.2
openpyxl==3.1.2
python-docx==1.1.0
tiktoken==0.7.0
pytest==8.2.0
pytest-asyncio==0.23.6
httpx==0.27.0            # for TestClient
```

---

## 7. Test Strategy

### 7.1 Unit Tests

#### `tests/test_parser.py`
| Test | Input | Expected |
|---|---|---|
| `test_parse_pdf` | `rfp_award_letter.pdf` | Returns list of `ParsedChunk`, each has `page_number >= 1` |
| `test_parse_csv` | `Recent Purchase History.csv` | Each row mapped to string chunk with `row_index` |
| `test_parse_xlsx` | `Cloud RFP Bidder Questions.xlsx` | Multi-sheet: chunks include sheet name in metadata |
| `test_parse_txt` | small `.txt` fixture | Returns single raw-text block |
| `test_unsupported_format` | `.zip` file | Raises `ValueError` |

#### `tests/test_chunker.py`
| Test | Input | Expected |
|---|---|---|
| `test_chunk_size_respected` | 3000-token text | All chunks ≤ 900 tokens |
| `test_overlap_present` | 2000-token text | Consecutive chunk pairs share ≥ 100 overlapping tokens |
| `test_small_doc_single_chunk` | 200-token text | Returns exactly 1 chunk |

#### `tests/test_embedder.py`
| Test | Input | Expected |
|---|---|---|
| `test_embedding_dimension` | list of 3 strings | Each vector has length 384 |
| `test_embedding_consistency` | same string twice | Both vectors are identical |
| `test_batch_embedding` | 100 strings | Returns 100 vectors, no error |

#### `tests/test_agents.py`
| Test | Agent | Mock | Expected |
|---|---|---|---|
| `test_retrieval_returns_chunks` | Retrieval | Mock Qdrant search | Returns sorted list with scores |
| `test_reasoning_no_context` | Reasoning | Mock Groq response | Returns "Information not found in context." |
| `test_reasoning_with_context` | Reasoning | Mock Groq w/ citations | Response includes `[filename, Page N]` pattern |
| `test_validator_passes_cited` | Validator | Cited draft answer | `validation_passed=True`, `citations_count >= 1` |
| `test_validator_fails_uncited` | Validator | Draft with no citations | `validation_passed=False` |
| `test_controller_mode_detection` | Controller | All agents mocked | "compare" keyword → mode=`comparative` |

---

### 7.2 Integration Tests

#### `tests/test_ingestion_api.py`
| Test | Method | Expected |
|---|---|---|
| `test_upload_pdf_returns_201` | `POST /documents` + `rfp_award_letter.pdf` | Status 201, `chunks_stored > 0` |
| `test_upload_csv_returns_201` | `POST /documents` + `Recent Purchase History.csv` | Status 201, `chunks_stored > 0` |
| `test_upload_xlsx_returns_201` | `POST /documents` + `Cloud RFP Bidder Questions.xlsx` | Status 201 |
| `test_upload_oversized_file` | `POST /documents` + synthetic 30MB file | Status 413 |
| `test_upload_unsupported_type` | `POST /documents` + `.zip` file | Status 422 |
| `test_upload_with_vendor_tag` | `POST /documents` + PDF + `vendor_name=AWS` | Response includes `vendor_name=AWS` |

#### `tests/test_chat_api.py`
| Test | Query | Expected |
|---|---|---|
| `test_extractive_query` | "What are the mandatory requirements?" | `mode=extractive`, answer non-empty, `sources` non-empty |
| `test_comparative_query` | "Compare annual pricing" | `mode=comparative`, answer contains Markdown table |
| `test_empty_db_returns_not_found` | any query on empty collection | answer == "Information not found in context." |
| `test_vendor_filter` | query + `vendor_filter=["AWS"]` | all `sources[*].vendor_name == "AWS"` |
| `test_citations_in_response` | pricing query | `citations_count >= 1`, answer contains `[` citation brackets `]` |
| `test_validation_flag_present` | any query | `validation_passed` boolean always present in response |

---

### 7.3 Test Data Mapping

| Test Scenario | Test File | Purpose |
|---|---|---|
| Ingestion — PDF | `testdata/rfp_award_letter.pdf` | Single-page PDF baseline |
| Ingestion — AWS proposal | `testdata/rfp_response_AWS/AWS Response...RFP_Public.pdf` | Multi-page PDF with structured text |
| Ingestion — Pricing CSV | `testdata/rfp_documents/Recent Purchase History.csv` | CSV row-chunking |
| Ingestion — Requirements XLSX | `testdata/rfp_documents/Cloud RFP Bidder Questions.xlsx` | Multi-column compliance rows |
| Comparative pricing query | AWS + Oracle + Google pricing PDFs | Cross-vendor comparison answer |
| Contract clause query | `testdata/rfp_response_Oracle/Oracle General Terms...pdf` | Contract extraction |

---

### 7.4 Test Execution

```bash
# All unit tests (no network calls; all external services mocked)
pytest tests/ -v --ignore=tests/test_ingestion_api.py --ignore=tests/test_chat_api.py

# Integration tests (requires running docker-compose stack)
docker compose up -d
pytest tests/test_ingestion_api.py tests/test_chat_api.py -v

# Full suite
pytest tests/ -v
```

---

## 8. Implementation Sequence

The following order minimizes blocked work and enables incremental testing:

| Phase | Tasks | Deliverable |
|---|---|---|
| **Phase 1** | `config.py`, `requirements.txt`, `.env.example`, `Dockerfile`, `docker-compose.yml` | Runnable empty FastAPI app + Qdrant |
| **Phase 2** | `ingestion/parser.py`, `ingestion/chunker.py`, `test_parser.py`, `test_chunker.py` | Parsing + chunking verified locally |
| **Phase 3** | `ingestion/embedder.py`, `vector_store/qdrant_client.py`, `test_embedder.py` | Embeddings stored in Qdrant |
| **Phase 4** | `api/documents.py` (POST /documents), `schemas/document.py`, `test_ingestion_api.py` | Ingestion endpoint passes all tests |
| **Phase 5** | `agents/retrieval.py`, `agents/reasoning.py`, `agents/extractor.py` | LLM reasoning over retrieved chunks |
| **Phase 6** | `agents/validator.py`, `agents/controller.py` | Full multi-agent pipeline end-to-end |
| **Phase 7** | `api/chat.py` (POST /chat), `schemas/chat.py`, `test_chat_api.py`, `test_agents.py` | Chat endpoint passes all tests |
| **Phase 8** | `GET /health`, README.md, final integration test run | Docs + acceptance criteria verified |

---

## 9. Acceptance Criteria Verification

| Scenario | How Verified |
|---|---|
| **Scenario 1** — Upload PDF + CSV → 201 + vectors stored | `test_upload_pdf_returns_201`, `test_upload_csv_returns_201` |
| **Scenario 2** — "Compare annual pricing" → Markdown table with correct values | `test_comparative_query`; manually run against AWS + Oracle + Google pricing PDFs |
| **Scenario 3** — Every factual claim has a source citation | `test_citations_in_response`; validator agent enforces citation pattern |
| **Hallucination guard** — no-context query returns sentinel string | `test_empty_db_returns_not_found` |
| **Docker reproducibility** — `docker compose up --build` succeeds cleanly | Dockerfile + compose verified in Phase 1 |

---

## 10. Key Design Decisions

1. **Groq instead of OpenAI**: The `openai` Python SDK is used with `base_url=https://api.groq.com/openai/v1` and the Groq API key. This is the officially supported pattern — no custom HTTP code needed.

2. **Local HuggingFace embeddings**: `sentence-transformers/all-MiniLM-L6-v2` runs fully locally (384 dimensions, ~22M params). No external embedding API calls. Model weights cached in a Docker named volume to avoid re-download on restart.

3. **Idempotent ingestion**: Chunk point IDs are derived from `sha256(filename + chunk_index)` so re-uploading the same file overwrites rather than duplicates.

4. **Citation validation via regex, not LLM**: The Validation Agent uses regex pattern matching against retrieved chunk metadata rather than a second LLM call — faster, cheaper, and deterministic.

5. **DOCX support**: `python-docx` added beyond the spec's stated formats because the testdata folder contains `.docx` RFP source documents that are valuable for requirements extraction.
