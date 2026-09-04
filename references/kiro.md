# Kiro Session Log
## Edureka Capstone B12 — Procurement & Vendor Evaluation Assistant

**Project path:** `c:\BASAK\Codebase\github_repos\edureka_capstone_b12_jkbasak\procurement-agent`  
**Session date:** 2026-09-03  
**Role:** Data Engineer

---

## Session Overview

Built a complete AI-powered Procurement & Vendor Evaluation Assistant from scratch,
migrated the vector database from Qdrant to ChromaDB, and documented the full
local-run setup. All work is inside the `procurement-agent/` subdirectory.

---

## Activity 1 — Review Specs and Create PLAN.md

**Prompt:** Review `specs/procurement-agent.md` and `specs/capstone_guidelines.md`,
then create `PLAN.md` with technical implementation details.

**Files read:**
- `procurement-agent/specs/procurement-agent.md`
- `procurement-agent/specs/capstone_guidelines.md`
- `procurement-agent/.env`
- `procurement-agent/testdata/` (directory listing)

**Output:** `procurement-agent/PLAN.md`

**Key decisions documented in PLAN.md:**
- Use Groq API (`llama-3.3-70b-versatile`) via OpenAI-compatible SDK
- Use HuggingFace `sentence-transformers/all-MiniLM-L6-v2` for local embeddings (384 dims)
- Qdrant as vector DB (later replaced with ChromaDB — see Activity 7)
- 8-phase implementation sequence
- Full API schema: `POST /documents`, `POST /chat`, `POST /chat/compliance`, `GET /health`
- 102 automated tests across 6 test files

---

## Activity 2 — Phase 1: Project Scaffold

**Files created:**
| File | Purpose |
|---|---|
| `app/__init__.py` | Package marker |
| `app/api/__init__.py` | Package marker |
| `app/agents/__init__.py` | Package marker |
| `app/ingestion/__init__.py` | Package marker |
| `app/vector_store/__init__.py` | Package marker |
| `app/schemas/__init__.py` | Package marker |
| `tests/__init__.py` | Package marker |
| `app/config.py` | Pydantic Settings loading `.env`; `get_settings()` singleton via `lru_cache` |
| `requirements.txt` | Pinned Python dependencies |
| `.env.example` | Template (no secrets) |
| `Dockerfile` | `python:3.11-slim`, installs requirements, runs uvicorn |
| `docker-compose.yml` | FastAPI + Qdrant services with healthchecks |
| `app/main.py` | FastAPI app, lifespan startup, CORS, `/health` endpoint, router registration |

---

## Activity 3 — Phase 2: Ingestion Layer

**Files created:**
| File | Purpose |
|---|---|
| `app/ingestion/parser.py` | Multi-format parser: PDF (PyMuPDF), TXT, CSV (pandas), XLSX (openpyxl), DOCX (python-docx). Returns `list[ParsedChunk]` |
| `app/ingestion/chunker.py` | Token-aware sliding-window chunker with tiktoken. `chunk_parsed_document()` → `list[TextChunk]`. Configurable `chunk_size` and `chunk_overlap` |

**Key design:**
- `ParsedChunk` dataclass carries raw text + provenance (source_file, page_number, row_index, sheet_name, vendor_name, doc_category)
- `TextChunk` extends with chunk_index, total_chunks, token_count
- Sentence-boundary splitting prevents mid-sentence cuts
- Overlap implemented via `_trim_overlap()` keeping suffix of window ≤ overlap_tokens

---

## Activity 4 — Phase 3: Embeddings & Vector Store

**Files created:**
| File | Purpose |
|---|---|
| `app/ingestion/embedder.py` | HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, batched encoding (32/batch), normalised vectors, `lru_cache` singleton. `embed_texts()` and `embed_query()` |
| `app/vector_store/qdrant_client.py` | `QdrantManager`: `ensure_collection`, `upsert_chunks` (sha256 idempotent IDs), `search` with vendor filter, `delete_by_source`, `collection_info` |

---

## Activity 5 — Phase 4: Pydantic Schemas

**Files created:**
| File | Purpose |
|---|---|
| `app/schemas/document.py` | `DocumentUploadResponse`, `ChunkMetadata`, `DeleteDocumentResponse` |
| `app/schemas/chat.py` | `ChatRequest` (with validators), `SourceReference`, `ChatResponse` |

---

## Activity 6 — Phase 5: Agents

**Files created:**
| File | Agent | Responsibility |
|---|---|---|
| `app/agents/retrieval.py` | Retrieval Agent | Embeds query, pulls top-K from vector DB, returns `list[RetrievedChunk]` with `citation_label()` and `short_excerpt()` helpers |
| `app/agents/extractor.py` | Requirement Extraction Agent | Parses Excel/CSV checklists; `extract_requirements()` with heuristic column detection; `build_compliance_matrix()` per (requirement × vendor); `format_compliance_table()` Markdown renderer |
| `app/agents/reasoning.py` | Reasoning Agent | Groq LLM via `openai` SDK with `base_url` override; strict grounding system prompt; `NOT_FOUND_SENTINEL` constant; `_build_context_block()` renders numbered context |
| `app/agents/validator.py` | Validation Agent | Regex citation check (`[filename, Page N]`); per-paragraph coverage; single rewrite attempt; `_is_exempt()` skips headers/tables/bullets |
| `app/agents/controller.py` | Agent Controller | Full pipeline: classify_mode → retrieve → reason → validate → `ChatResponse`; `classify_mode()` keyword regex for comparative vs extractive |

---

## Activity 7 — Phase 6: API Routers

**Files created:**
| File | Endpoints |
|---|---|
| `app/api/documents.py` | `POST /documents` (parse→chunk→embed→upsert, 201/422/413), `DELETE /documents/{filename}`, `GET /documents` |
| `app/api/chat.py` | `POST /chat` (full agent pipeline), `POST /chat/compliance` (requirements file → compliance matrix) |
| `app/dependencies.py` | `get_settings_dep()`, `get_chroma()` FastAPI dependency injectors |

---

## Activity 8 — Phase 7: Test Suite (102 tests)

**Files created:**
| File | Tests | What is tested |
|---|---|---|
| `tests/conftest.py` | — | Fixtures: in-memory ChromaDB EphemeralClient, zero-vector mock embeddings, canned LLM mock, TestClient, sample bytes (PDF/TXT/CSV/XLSX) |
| `tests/test_parser.py` | 18 | TXT (latin-1 fallback), CSV (row format, row_index), XLSX (multi-sheet), PDF (page_number), unsupported extension |
| `tests/test_chunker.py` | 14 | Size ceiling, overlap presence, single-chunk small docs, contiguous chunk_index, metadata propagation |
| `tests/test_embedder.py` | 12 | Vector dimension (384), empty input, non-list raises, batch of 100, consistency |
| `tests/test_agents.py` | 28 | All 5 agents: retrieval hits, citation labels, reasoning sentinel, LLM call args, validation pass/fail/rewrite, compliance table, controller mode detection |
| `tests/test_ingestion_api.py` | 12 | POST/DELETE/GET /documents; 201/422/413 status codes; idempotent re-upload |
| `tests/test_chat_api.py` | 18 | Extractive/comparative mode, citation in answer, hallucination guard (empty DB → sentinel), vendor filter, compliance endpoint |
| `pytest.ini` | — | testpaths, asyncio_mode=auto, warning filters |

---

## Activity 9 — Phase 8: README.md

**File created:** `procurement-agent/README.md`

Contents:
- Architecture diagram (ASCII)
- Agent roles table
- Quick Start (Docker + local Python)
- Configuration table (all `.env` variables)
- API Reference with `curl` examples for all endpoints
- Sample Workflows using the actual Michigan DTMB RFP testdata filenames
- Project structure tree
- Running Tests section
- Design Decisions (Groq/OpenAI SDK, local embeddings, idempotent IDs, regex validation, hallucination guard)

---

## Activity 10 — Local Run (no Docker)

**Question:** How to run without Docker?

**Answer summary:**

```powershell
# Option A: Qdrant (at that point) via Docker single container
docker run -p 6333:6333 qdrant/qdrant:v1.9.2

# Create venv and install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Change QDRANT_HOST=localhost in .env
# Run API
uvicorn app.main:app --reload --port 8000
```

---

## Activity 11 — Migrate Vector DB: Qdrant → ChromaDB

**Reason:** ChromaDB is pure Python, runs in-process, requires no server or Docker.

### Files created
| File | Purpose |
|---|---|
| `app/vector_store/chroma_client.py` | `ChromaManager` class with identical interface to `QdrantManager`. Uses `chromadb.PersistentClient` (or injected `EphemeralClient` for tests). Cosine distance via `hnsw:space=cosine`. SHA256 hex IDs. `None`→sentinel value mapping for Chroma metadata constraints. Cosine distance→similarity score conversion: `score = 1 - distance/2` |

### Files modified
| File | Change |
|---|---|
| `requirements.txt` | `qdrant-client==1.9.1` removed → `chromadb==0.5.3` added |
| `app/config.py` | `qdrant_host/port/collection` → `chroma_persist_dir=./chroma_db`, `chroma_collection` |
| `app/main.py` | `get_chroma_manager` import; lifespan logs Chroma path; `/health` returns `vector_db_type: chromadb` and `persist_dir` |
| `app/api/documents.py` | All `get_qdrant_manager()` → `get_chroma_manager()`; `list_documents` uses `chroma.list_sources()` (no Qdrant scroll needed) |
| `app/api/chat.py` | Docstrings updated |
| `app/agents/retrieval.py` | Imports `get_chroma_manager`; `__init__` parameter renamed `chroma_manager=` |
| `app/agents/controller.py` | Docstring updated |
| `app/dependencies.py` | `get_chroma()` wraps `get_chroma_manager()` |
| `.env` | Qdrant vars removed; `CHROMA_PERSIST_DIR=./chroma_db` + `CHROMA_COLLECTION=procurement_docs` added |
| `.env.example` | Same |
| `docker-compose.yml` | Qdrant service + `qdrant_data` volume removed; single `api` service with `chroma_data` named volume |
| `tests/conftest.py` | `InMemoryQdrantManager` dict mock replaced with real `chromadb.EphemeralClient` wrapped in `ChromaManager`. Session fixture `mock_chroma`. Module singleton patched via `cc_module.get_chroma_manager = lambda: mock_chroma` |
| `tests/test_agents.py` | 4 `RetrievalAgent(qdrant_manager=...)` → `chroma_manager=`; docstrings updated |
| `tests/test_chat_api.py` | Hallucination guard test uses `EphemeralClient` + `ChromaManager` instead of custom dict mock |
| `tests/test_ingestion_api.py` | Docstrings updated |
| `README.md` | Architecture diagram, quickstart (local run now 2 commands), config table, `/health` response, structure tree, design decisions — all updated for ChromaDB |

### ChromaDB metadata constraint
ChromaDB does not accept `None` values in metadata. Mapping applied:
- `None` string fields → `""` (empty string)
- `None` int fields (page_number, row_index) → `-1`
- Reverse mapping applied in `_from_metadata()` when returning search results

### New local run (after migration — no Docker at all)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# Edit .env: set GROQ_API_KEY=gsk_...
uvicorn app.main:app --reload --port 8000
```
ChromaDB creates `./chroma_db/` directory automatically on first startup.

---

## Final Project Structure

```
procurement-agent/
├── .env                          # Runtime secrets
├── .env.example                  # Template
├── docker-compose.yml            # Single API service (no Qdrant)
├── Dockerfile                    # python:3.11-slim
├── requirements.txt              # chromadb==0.5.3 (no qdrant-client)
├── pytest.ini
├── PLAN.md
├── README.md
│
├── app/
│   ├── main.py                   # FastAPI + lifespan + /health
│   ├── config.py                 # CHROMA_PERSIST_DIR, CHROMA_COLLECTION
│   ├── dependencies.py           # get_settings_dep, get_chroma
│   ├── api/
│   │   ├── documents.py          # POST/GET/DELETE /documents
│   │   └── chat.py               # POST /chat + POST /chat/compliance
│   ├── agents/
│   │   ├── controller.py         # Pipeline orchestration + mode detection
│   │   ├── extractor.py          # Requirement Extraction + compliance matrix
│   │   ├── retrieval.py          # ChromaDB search wrapper
│   │   ├── reasoning.py          # Groq LLM (openai SDK, strict grounding prompt)
│   │   └── validator.py          # Citation regex check + rewrite trigger
│   ├── ingestion/
│   │   ├── parser.py             # PDF/TXT/CSV/XLSX/DOCX → ParsedChunk
│   │   ├── chunker.py            # Sliding-window token chunker → TextChunk
│   │   └── embedder.py           # all-MiniLM-L6-v2, batched, normalised
│   ├── vector_store/
│   │   ├── chroma_client.py      # ChromaManager (active)
│   │   └── qdrant_client.py      # Archived — not imported
│   └── schemas/
│       ├── document.py
│       └── chat.py
│
└── tests/                        # 102 tests, all run offline
    ├── conftest.py               # EphemeralClient + zero-vector mocks
    ├── test_parser.py            # 18 tests
    ├── test_chunker.py           # 14 tests
    ├── test_embedder.py          # 12 tests
    ├── test_agents.py            # 28 tests
    ├── test_ingestion_api.py     # 12 tests
    └── test_chat_api.py          # 18 tests
```

---

## Key Environment Variables Reference

| Variable | Value | Notes |
|---|---|---|
| `GROQ_API_KEY` | `gsk_...` | Groq cloud LLM key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local HuggingFace model |
| `CHUNK_SIZE` | `900` | Tokens per chunk |
| `CHUNK_OVERLAP` | `150` | ~17% overlap |
| `TOP_K` | `6` | Chunks per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local: relative path; Docker: `/app/chroma_db` |
| `CHROMA_COLLECTION` | `procurement_docs` | ChromaDB collection name |
| `MAX_UPLOAD_MB` | `25` | Upload size limit |
| `LOG_LEVEL` | `INFO` | Python log level |

---

## Test Data (Michigan DTMB RFP)

```
testdata/
├── rfp_award_letter.pdf
├── rfp_documents/
│   ├── Cloud RFP Bidder Questions.xlsx       ← compliance checklist
│   ├── Cloud Services RFP 250000000414 Questions Round 2.xlsx
│   ├── Recent Purchase History.csv           ← pricing/purchase data
│   ├── RFP 250000000414 - Cloud Services 12.19.24 version.docx
│   └── RFP 250000000414 - Cloud Services.docx
├── rfp_response_AWS/
│   ├── AWS Response ... RFP Schedule B Pricing.pdf   ← pricing
│   └── AWS Response ... RFP_Public.pdf               ← proposal
├── rfp_response_Google/
│   ├── Appendix 1 - Schedule A - Technical Req 6.0 Answer.pdf
│   ├── Google Cover Letter.pdf
│   ├── Google RFP ... Cloud Services ... Public.pdf
│   └── SecOps Services Schedule Template ... .pdf
└── rfp_response_Oracle/
    ├── Oracle General Terms and Conditions ... .pdf
    ├── RFP ... Oracle Redacted 1/2/3 of 3 PUBLIC COPY.pdf
    ├── Schedule B Pricing-Oracle.pdf                  ← pricing
    └── (+ 9 other Oracle docs)
```
