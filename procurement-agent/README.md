# Procurement & Vendor Evaluation Assistant

An AI-powered multi-agent system that automates RFP analysis, vendor comparison,
and compliance validation for procurement teams.

Upload vendor proposals, pricing sheets, and requirements documents.
Ask natural-language questions. Get grounded, cited answers — no hallucinations.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Sample Workflows](#sample-workflows)
6. [Project Structure](#project-structure)
7. [Running Tests](#running-tests)
8. [Design Decisions](#design-decisions)

---

## Architecture

```
User / curl / UI
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI (port 8000)                       │
│   POST /documents    POST /chat    POST /chat/compliance     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Agent Controller │
                  └────────┬────────┘
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  Retrieval   │  │  Reasoning   │  │  Validation  │
  │    Agent     │  │    Agent     │  │    Agent     │
  └──────┬───────┘  └──────┬───────┘  └──────────────┘
         │                 │
         ▼                 ▼
  ┌──────────────┐  ┌──────────────┐
  │  Qdrant      │  │  Groq LLM    │
  │  Vector DB   │  │  (llama-3.3) │
  └──────────────┘  └──────────────┘
```

### Agent Roles

| Agent | Responsibility |
|---|---|
| **Requirement Extraction** | Parses Excel/CSV compliance checklists; builds Pass/Fail/Partially Met matrices |
| **Retrieval** | Embeds query with `all-MiniLM-L6-v2`; pulls top-K chunks from Qdrant |
| **Reasoning** | Calls Groq LLM with strict grounding prompt; produces Markdown answers with citations |
| **Validation** | Checks every factual claim has a `[filename, Page N]` citation; triggers one rewrite if not |
| **Controller** | Orchestrates the pipeline; classifies queries as extractive vs comparative |

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) ≥ 24
- A [Groq API key](https://console.groq.com/) (free tier available)

### 1. Clone and configure

```bash
git clone <repo-url>
cd procurement-agent

# Copy the example env file and add your Groq key
cp .env.example .env
# Edit .env: set GROQ_API_KEY=gsk_...
```

### 2. Start the stack

```bash
docker compose up --build
```

On first start, Docker will:
- Pull the `qdrant/qdrant:v1.9.2` image
- Build the FastAPI image (`python:3.11-slim`)
- Download the `all-MiniLM-L6-v2` embedding model (~90 MB, cached in a named volume)

The API is ready when you see:
```
procurement_api  | INFO:     Application startup complete.
```

### 3. Open the interactive docs

```
http://localhost:8000/docs
```

---

## Configuration

All settings are driven by the `.env` file.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq cloud API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `900` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap tokens between chunks (~17%) |
| `TOP_K` | `6` | Chunks retrieved per query |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname (use `localhost` outside Docker) |
| `QDRANT_PORT` | `6333` | Qdrant REST port |
| `QDRANT_COLLECTION` | `procurement_docs` | Vector collection name |
| `MAX_UPLOAD_MB` | `25` | Maximum upload file size |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## API Reference

### `POST /documents` — Ingest a document

Upload a document to parse, chunk, embed, and store in Qdrant.

**Supported formats:** `.pdf` `.txt` `.csv` `.xlsx` `.docx`

```bash
# Ingest an AWS proposal PDF
curl -X POST http://localhost:8000/documents \
  -F "file=@testdata/rfp_response_AWS/AWS Response to Michigan DTMB Cloud Services RFP_Public.pdf" \
  -F "vendor_name=AWS" \
  -F "doc_category=proposal"

# Ingest a pricing CSV
curl -X POST http://localhost:8000/documents \
  -F "file=@testdata/rfp_documents/Recent Purchase History.csv" \
  -F "doc_category=pricing"

# Ingest an Oracle pricing PDF
curl -X POST http://localhost:8000/documents \
  -F "file=@testdata/rfp_response_Oracle/Schedule B Pricing-Oracle.pdf" \
  -F "vendor_name=Oracle" \
  -F "doc_category=pricing"

# Ingest a Google response
curl -X POST http://localhost:8000/documents \
  -F "file=@testdata/rfp_response_Google/Google RFP 250000000414 - Cloud Services 12.19.24 version (1)_Public.pdf" \
  -F "vendor_name=Google" \
  -F "doc_category=proposal"
```

**Response (201 Created):**
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

---

### `POST /chat` — Ask a question

```bash
# Extractive — single fact lookup
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Which vendor satisfies all mandatory requirements?"}'

# Comparative — cross-vendor analysis
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare annual pricing across AWS, Google, and Oracle"}'

# With vendor filter
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Does AWS hold FedRAMP authorization?",
    "vendor_filter": ["AWS"],
    "mode": "extractive"
  }'

# Contract clause search
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Which contract contains an auto-renewal clause?"}'
```

**Response (200 OK):**
```json
{
  "answer": "## Pricing Comparison\n\n| Vendor | Annual Cost | ...\n\nAWS offers competitive pricing [AWS Schedule B Pricing.pdf, Page 3].",
  "mode": "comparative",
  "sources": [
    {
      "source_file": "AWS Response to Michigan DTMB Cloud Services RFP Schedule B Pricing.pdf",
      "vendor_name": "AWS",
      "page_number": 3,
      "chunk_index": 12,
      "score": 0.91,
      "text_excerpt": "Annual cloud service pricing for FY2025..."
    }
  ],
  "validation_passed": true,
  "citations_count": 3
}
```

**No-context fallback:**
```json
{
  "answer": "Information not found in context.",
  "mode": "extractive",
  "sources": [],
  "validation_passed": true,
  "citations_count": 0
}
```

---

### `POST /chat/compliance` — Compliance matrix

```bash
curl -X POST http://localhost:8000/chat/compliance \
  -F "file=@testdata/rfp_documents/Cloud RFP Bidder Questions.xlsx" \
  -F "vendors=AWS,Google,Oracle"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "requirements_count": 24,
  "vendors": ["AWS", "Google", "Oracle"],
  "compliance_table": "| Req ID | Description | AWS | Google | Oracle |\n|...",
  "verdicts": [
    {
      "requirement_id": "REQ-001",
      "vendor_name": "AWS",
      "status": "Pass",
      "evidence": "AWS confirms 99.99% uptime SLA...",
      "citation": "[AWS_Proposal.pdf, Page 3]"
    }
  ]
}
```

---

### `GET /documents` — List indexed documents

```bash
curl http://localhost:8000/documents
```

### `DELETE /documents/{filename}` — Remove a document

```bash
curl -X DELETE "http://localhost:8000/documents/Schedule%20B%20Pricing-Oracle.pdf"
```

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
```

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

## Sample Workflows

### Workflow 1 — Ingest all Michigan DTMB RFP documents and compare vendors

```bash
# 1. Ingest the RFP requirements
curl -X POST http://localhost:8000/documents \
  -F "file=@testdata/rfp_documents/RFP 250000000414 - Cloud Services 12.19.24 version.docx" \
  -F "doc_category=requirements"

# 2. Ingest all three vendor proposals
for vendor in AWS Google Oracle; do
  echo "Ingesting $vendor documents..."
done

# (Use the curl commands from the API Reference section above)

# 3. Ask comparison question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare annual pricing across AWS, Google, and Oracle"}'

# 4. Check mandatory requirement compliance
curl -X POST http://localhost:8000/chat/compliance \
  -F "file=@testdata/rfp_documents/Cloud RFP Bidder Questions.xlsx" \
  -F "vendors=AWS,Google,Oracle"

# 5. Identify missing information
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Identify missing information in the vendor responses"}'
```

### Workflow 2 — Contract clause analysis

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which contract contains an auto-renewal clause?",
    "vendor_filter": ["Oracle"]
  }'
```

---

## Project Structure

```
procurement-agent/
├── .env                          # Runtime secrets (not committed)
├── .env.example                  # Template — copy to .env
├── docker-compose.yml            # FastAPI + Qdrant services
├── Dockerfile                    # FastAPI image build
├── requirements.txt              # Pinned Python dependencies
├── pytest.ini                    # pytest configuration
├── PLAN.md                       # Technical implementation plan
├── README.md                     # This file
│
├── app/
│   ├── main.py                   # FastAPI app, lifespan, router registration
│   ├── config.py                 # Pydantic Settings (reads .env)
│   ├── dependencies.py           # Shared FastAPI dependency injectors
│   │
│   ├── api/
│   │   ├── documents.py          # POST/GET/DELETE /documents
│   │   └── chat.py               # POST /chat, POST /chat/compliance
│   │
│   ├── agents/
│   │   ├── controller.py         # Agent Controller — full pipeline orchestration
│   │   ├── extractor.py          # Requirement Extraction Agent
│   │   ├── retrieval.py          # Retrieval Agent (Qdrant + embeddings)
│   │   ├── reasoning.py          # Reasoning Agent (Groq LLM)
│   │   └── validator.py          # Validation Agent (citation checker)
│   │
│   ├── ingestion/
│   │   ├── parser.py             # Multi-format parser (PDF/TXT/CSV/XLSX/DOCX)
│   │   ├── chunker.py            # Token-aware sliding-window chunker
│   │   └── embedder.py           # HuggingFace sentence-transformers wrapper
│   │
│   ├── vector_store/
│   │   └── qdrant_client.py      # Qdrant collection management + upsert/search
│   │
│   └── schemas/
│       ├── document.py           # Ingestion request/response models
│       └── chat.py               # Chat request/response models
│
├── tests/
│   ├── conftest.py               # Fixtures: in-memory Qdrant, mocked LLM, TestClient
│   ├── test_parser.py            # Parser unit tests (18 tests)
│   ├── test_chunker.py           # Chunker unit tests (14 tests)
│   ├── test_embedder.py          # Embedder unit tests (12 tests)
│   ├── test_agents.py            # Agent unit tests (28 tests)
│   ├── test_ingestion_api.py     # /documents integration tests (12 tests)
│   └── test_chat_api.py          # /chat integration tests (18 tests)
│
├── specs/
│   ├── procurement-agent.md      # System specification
│   └── capstone_guidelines.md    # Capstone requirements
│
└── testdata/                     # Michigan DTMB RFP sample documents
    ├── rfp_award_letter.pdf
    ├── rfp_documents/            # RFP requirements + bidder questions
    ├── rfp_response_AWS/         # AWS proposal + pricing
    ├── rfp_response_Google/      # Google proposal + appendices
    └── rfp_response_Oracle/      # Oracle proposal + pricing + terms
```

---

## Running Tests

Tests run entirely offline — no Docker, no Groq API, no model download needed.
All external dependencies are mocked.

```bash
# Install dependencies (one-time)
pip install -r requirements.txt

# Run all unit tests (fast, fully mocked)
pytest tests/ -v \
  --ignore=tests/test_ingestion_api.py \
  --ignore=tests/test_chat_api.py

# Run integration tests (requires docker compose stack running)
docker compose up -d
pytest tests/test_ingestion_api.py tests/test_chat_api.py -v

# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_agents.py -v

# Run a specific test
pytest tests/test_agents.py::TestAgentController::test_classify_mode_comparative -v
```

**Test coverage by module:**

| Test File | Module Tested | Tests |
|---|---|---|
| `test_parser.py` | `ingestion/parser.py` | 18 |
| `test_chunker.py` | `ingestion/chunker.py` | 14 |
| `test_embedder.py` | `ingestion/embedder.py` | 12 |
| `test_agents.py` | All 5 agents | 28 |
| `test_ingestion_api.py` | `api/documents.py` | 12 |
| `test_chat_api.py` | `api/chat.py` | 18 |
| **Total** | | **102** |

---

## Design Decisions

### Groq API via OpenAI SDK

Groq exposes an OpenAI-compatible REST API. The `openai` Python SDK is used with
`base_url="https://api.groq.com/openai/v1"` and the Groq key — no custom HTTP
client needed, and the code can switch to any OpenAI-compatible provider by
changing two `.env` variables.

### Local HuggingFace embeddings

`sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional vectors and runs
fully on CPU in under 100 ms per batch. Weights are cached in a Docker named volume
so the ~90 MB download happens only on the first `docker compose up`.

### Idempotent ingestion

Each chunk's Qdrant point ID is derived from `sha256(filename + chunk_index)`.
Re-uploading the same file overwrites the same point IDs, preventing duplicates
without needing a separate deduplication step.

### Citation validation by regex, not LLM

The Validation Agent uses a regex pattern (`[filename, Page N]`) against the
retrieved chunk metadata rather than a second LLM call. This is faster, cheaper,
deterministic, and avoids recursive validation loops.

### Hallucination guard

The Reasoning Agent's system prompt forbids the LLM from answering outside the
provided context. If the vector search returns zero chunks, the controller
short-circuits and returns `"Information not found in context."` before the LLM
is ever called.

### DOCX support beyond spec

The specification lists PDF, TXT, CSV, XLSX. DOCX was added because the testdata
folder contains `.docx` RFP source documents (`RFP 250000000414 - Cloud Services.docx`)
that are essential for full requirements extraction.
