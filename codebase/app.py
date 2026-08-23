"""
Procurement Vendor Comparison — Multi-Agent Knowledge & Decision Support System
================================================================================
FastAPI backend implementing a Retrieval-Augmented, multi-agent pipeline for
comparing vendor proposals, pricing, requirements, and contracts.

Agents:
  1. RequirementExtractionAgent  - parses Requirements Excel into structured items
  2. RetrievalAgent              - semantic search over the vector store
  3. ComparisonReasoningAgent    - Groq LLM grounded reasoning w/ citations
  4. ValidationAgent             - citation/hallucination checks, confidence score

Run:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Env vars (see .env.example):
  GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
"""

from __future__ import annotations

import io
import os
import re
import uuid
import json
import logging
import hashlib
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---- Optional heavy deps guarded so the module can be imported/tested
# without them installed (useful for lightweight unit tests / CI). -----------
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None

# ==============================================================================
# Configuration
# ==============================================================================

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "900"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K: int = int(os.getenv("TOP_K", "6"))
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("procurement_agent")

# ==============================================================================
# Domain Models
# ==============================================================================

class DocType(str, Enum):
    PROPOSAL = "proposal"        # PDF
    PRICING = "pricing"          # CSV
    REQUIREMENTS = "requirements"  # Excel
    CONTRACT = "contract"        # TXT / PDF

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    vendor: str
    doc_type: DocType
    filename: str
    text: str
    position: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Requirement(BaseModel):
    req_id: str
    description: str
    mandatory: bool
    category: Optional[str] = None
    source_doc: str

class Citation(BaseModel):
    chunk_id: str
    vendor: str
    doc_type: DocType
    filename: str
    snippet: str
    score: float

class ChatRequest(BaseModel):
    query: str
    vendor_filter: Optional[List[str]] = None
    top_k: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    agents_used: List[str]
    validation: Dict[str, Any]
    intent: str

class UploadResponse(BaseModel):
    doc_id: str
    vendor: str
    doc_type: DocType
    filename: str
    chunks_created: int
    requirements_extracted: Optional[int] = None

# ==============================================================================
# Document Loading & Normalization
# ==============================================================================

class DocumentProcessingError(Exception):
    pass

def load_pdf(raw: bytes) -> str:
    if PdfReader is None:
        raise DocumentProcessingError("pypdf not installed on server")
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"[page {i + 1}]\n{text}")
        return "\n\n".join(pages)
    except Exception as e:
        raise DocumentProcessingError(f"Failed to parse PDF: {e}")

def load_txt(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DocumentProcessingError("Could not decode text file")

def load_csv(raw: bytes) -> Tuple[str, pd.DataFrame]:
    try:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [str(c).strip() for c in df.columns]
        text_rows = [", ".join(f"{c}: {r[c]}" for c in df.columns) for _, r in df.iterrows()]
        return "\n".join(text_rows), df
    except Exception as e:
        raise DocumentProcessingError(f"Failed to parse CSV: {e}")

def load_excel(raw: bytes) -> Tuple[str, pd.DataFrame]:
    try:
        df = pd.read_excel(io.BytesIO(raw))
        df.columns = [str(c).strip() for c in df.columns]
        text_rows = [", ".join(f"{c}: {r[c]}" for c in df.columns) for _, r in df.iterrows()]
        return "\n".join(text_rows), df
    except Exception as e:
        raise DocumentProcessingError(f"Failed to parse Excel: {e}")

def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Simple recursive-ish sliding window chunker on characters,
    preferring paragraph boundaries."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)

    # apply overlap between adjacent chunks for continuity
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            prefix = chunks[i - 1][-overlap:] if overlap else ""
            overlapped.append((prefix + " " + c).strip())
    return overlapped

# ==============================================================================
# Embedding + Vector Store
# ==============================================================================

class VectorStore:
    """Wraps sentence-transformers embeddings with a FAISS index.
    Falls back to numpy brute-force cosine search if FAISS is unavailable."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._dim: Optional[int] = None
        self._index = None
        self._vectors: List[np.ndarray] = []
        self.chunks: List[Chunk] = []

    def _lazy_load_model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise DocumentProcessingError("sentence-transformers not installed")
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            if faiss is not None:
                self._index = faiss.IndexFlatIP(self._dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        self._lazy_load_model()
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def add(self, chunks: List[Chunk]):
        if not chunks:
            return
        vecs = self.embed([c.text for c in chunks])
        if faiss is not None:
            if self._index is None:
                self._lazy_load_model()
            self._index.add(vecs)
        else:
            self._vectors.extend(list(vecs))
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 6, vendor_filter: Optional[List[str]] = None) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []
        qvec = self.embed([query])[0]

        if faiss is not None and self._index is not None and self._index.ntotal > 0:
            k = min(top_k * 4, self._index.ntotal)  # over-fetch to allow post-filtering
            scores, idxs = self._index.search(np.expand_dims(qvec, 0), k)
            candidates = [(self.chunks[i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i != -1]
        else:
            sims = [float(np.dot(qvec, v)) for v in self._vectors]
            ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[: top_k * 4]
            candidates = [(self.chunks[i], sims[i]) for i in ranked]

        if vendor_filter:
            vf = {v.lower() for v in vendor_filter}
            candidates = [(c, s) for c, s in candidates if c.vendor.lower() in vf]

        return candidates[:top_k]

    def vendors(self) -> List[str]:
        return sorted({c.vendor for c in self.chunks})

    def doc_count_by_vendor(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for c in self.chunks:
            out.setdefault(c.vendor, {}).setdefault(c.doc_type.value, 0)
            out[c.vendor][c.doc_type.value] += 1
        return out

# Global in-memory stores (swap for a real DB / persistent vector store in production)
VECTOR_STORE = VectorStore()
REQUIREMENTS_STORE: List[Requirement] = []
DOCUMENT_REGISTRY: Dict[str, Dict[str, Any]] = {}

# ==============================================================================
# Groq LLM Client Wrapper
# ==============================================================================

class LLMClient:
    def __init__(self):
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            if Groq is None:
                raise DocumentProcessingError("groq SDK not installed")
            if not settings.GROQ_API_KEY:
                raise DocumentProcessingError("GROQ_API_KEY not configured")
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        return self._client

    def complete(self, system: str, user: str, temperature: float = 0.1, json_mode: bool = False) -> str:
        client = self._lazy_client()
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=1500,
            **kwargs,
        )
        return resp.choices[0].message.content

LLM = LLMClient()

# ==============================================================================
# AGENT 1 — Requirement Extraction Agent
# ==============================================================================

class RequirementExtractionAgent:
    """Parses a Requirements Excel sheet into structured Requirement objects.
    Expected (flexible) columns: ID/Requirement ID, Description, Mandatory (Y/N),
    Category. Falls back to heuristics if columns are named differently."""

    MANDATORY_TRUE = {"y", "yes", "true", "1", "mandatory", "required"}

    def run(self, df: pd.DataFrame, source_doc: str) -> List[Requirement]:
        cols = {c.lower().strip(): c for c in df.columns}

        def find_col(*candidates):
            for cand in candidates:
                for lc, orig in cols.items():
                    if cand in lc:
                        return orig
            return None

        id_col = find_col("requirement id", "req id", "id")
        desc_col = find_col("description", "requirement", "detail")
        mand_col = find_col("mandatory", "required", "priority")
        cat_col = find_col("category", "type", "group")

        if desc_col is None:
            raise DocumentProcessingError(
                "Could not locate a requirement description column in the Requirements Excel"
            )

        requirements = []
        for i, row in df.iterrows():
            desc = str(row[desc_col]).strip()
            if not desc or desc.lower() == "nan":
                continue
            mandatory = True
            if mand_col is not None:
                mandatory = str(row[mand_col]).strip().lower() in self.MANDATORY_TRUE
            req_id = str(row[id_col]).strip() if id_col is not None else f"REQ-{i+1:03d}"
            category = str(row[cat_col]).strip() if cat_col is not None else None
            requirements.append(
                Requirement(req_id=req_id, description=desc, mandatory=mandatory,
                            category=category, source_doc=source_doc)
            )
        logger.info(f"RequirementExtractionAgent: extracted {len(requirements)} requirements")
        return requirements

# ==============================================================================
# AGENT 2 — Retrieval Agent
# ==============================================================================

class RetrievalAgent:
    def __init__(self, store: VectorStore):
        self.store = store

    def run(self, query: str, top_k: int = None, vendor_filter: Optional[List[str]] = None) -> List[Tuple[Chunk, float]]:
        top_k = top_k or settings.TOP_K
        results = self.store.search(query, top_k=top_k, vendor_filter=vendor_filter)
        logger.info(f"RetrievalAgent: retrieved {len(results)} chunks for query='{query[:60]}...'")
        return results

# ==============================================================================
# AGENT 3 — Comparison / Reasoning Agent
# ==============================================================================

SYSTEM_PROMPT = """You are a meticulous procurement analyst assistant. You answer
questions ONLY using the evidence snippets provided to you. Every factual claim
you make MUST be traceable to a specific evidence snippet, cited inline as [S1],
[S2], etc., matching the snippet numbers given. If the evidence does not contain
enough information to answer confidently, say so explicitly and list what is
missing rather than guessing. Never invent vendor names, prices, or clauses that
are not present in the evidence. When comparing vendors, present the comparison
clearly (e.g., a short table or bullet list per vendor). Be concise and factual."""

class ComparisonReasoningAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _format_evidence(self, results: List[Tuple[Chunk, float]]) -> str:
        lines = []
        for i, (chunk, score) in enumerate(results, 1):
            lines.append(
                f"[S{i}] (vendor={chunk.vendor}, doc_type={chunk.doc_type.value}, "
                f"file={chunk.filename}, relevance={score:.2f})\n{chunk.text}"
            )
        return "\n\n".join(lines)

    def run(self, query: str, results: List[Tuple[Chunk, float]],
            requirements: Optional[List[Requirement]] = None) -> str:
        if not results:
            return ("I could not find any relevant evidence in the uploaded documents "
                    "to answer this question. Please confirm the relevant documents have "
                    "been uploaded.")

        evidence_block = self._format_evidence(results)
        req_block = ""
        if requirements:
            req_lines = [
                f"- {r.req_id} [{'MANDATORY' if r.mandatory else 'optional'}]"
                f"{f' ({r.category})' if r.category else ''}: {r.description}"
                for r in requirements
            ]
            req_block = "\n\nKNOWN REQUIREMENTS:\n" + "\n".join(req_lines)

        user_prompt = (
            f"QUESTION:\n{query}\n\nEVIDENCE SNIPPETS:\n{evidence_block}{req_block}\n\n"
            "Answer the question now, citing snippets like [S1]. If comparing vendors, "
            "organize the answer by vendor."
        )

        try:
            answer = self.llm.complete(SYSTEM_PROMPT, user_prompt, temperature=0.1)
        except DocumentProcessingError as e:
            logger.warning(f"LLM unavailable, falling back to extractive summary: {e}")
            answer = self._extractive_fallback(query, results)
        return answer

    def _extractive_fallback(self, query: str, results: List[Tuple[Chunk, float]]) -> str:
        """Degrades gracefully to a purely extractive answer if no LLM is configured,
        so the system still works without a GROQ_API_KEY."""
        lines = [f"(LLM not configured — showing top retrieved evidence for: '{query}')\n"]
        for i, (chunk, score) in enumerate(results, 1):
            lines.append(f"[S{i}] {chunk.vendor} / {chunk.doc_type.value} / {chunk.filename} "
                         f"(score={score:.2f}):\n{chunk.text[:400]}...\n")
        return "\n".join(lines)

# ==============================================================================
# AGENT 4 — Validation Agent
# ==============================================================================

class ValidationAgent:
    """Checks the reasoning agent's answer for grounding: every citation marker
    must correspond to a real snippet, and cited vendors/numbers should actually
    appear somewhere in the evidence text (lexical overlap heuristic)."""

    CITATION_RE = re.compile(r"\[S(\d+)\]")

    def run(self, answer: str, results: List[Tuple[Chunk, float]]) -> Dict[str, Any]:
        n_snippets = len(results)
        cited = {int(m) for m in self.CITATION_RE.findall(answer)}
        invalid_citations = sorted(c for c in cited if c < 1 or c > n_snippets)

        evidence_text = " ".join(c.text.lower() for c, _ in results)
        vendors_in_evidence = {c.vendor.lower() for c, _ in results}
        vendors_in_answer = set(re.findall(r"\b[A-Z][A-Za-z0-9&.\-]{2,}\b", answer))
        unverified_vendor_mentions = [
            v for v in vendors_in_answer
            if v.lower() not in vendors_in_evidence
            and v.lower() not in evidence_text
            and v.lower() not in {"the", "yes", "no", "mandatory", "s1", "s2", "s3"}
        ]

        has_citations = len(cited) > 0
        no_invalid = len(invalid_citations) == 0
        grounding_ratio = len(cited) / max(1, len(answer.split("."))) if has_citations else 0.0

        confidence = "low"
        if has_citations and no_invalid and n_snippets > 0:
            confidence = "high" if grounding_ratio > 0.15 else "medium"
        if n_snippets == 0:
            confidence = "unanswerable"

        flags = []
        if not has_citations and n_snippets > 0:
            flags.append("Answer contains no citations despite evidence being available.")
        if invalid_citations:
            flags.append(f"Answer references non-existent snippet(s): {invalid_citations}")
        if unverified_vendor_mentions[:3]:
            flags.append(f"Possible unverified names mentioned: {unverified_vendor_mentions[:3]}")

        return {
            "confidence": confidence,
            "citations_found": sorted(cited),
            "invalid_citations": invalid_citations,
            "snippets_available": n_snippets,
            "flags": flags,
            "passed": no_invalid and (has_citations or n_snippets == 0),
        }

# ==============================================================================
# ORCHESTRATOR — Planner Agent
# ==============================================================================

class Orchestrator:
    def __init__(self):
        self.retrieval_agent = RetrievalAgent(VECTOR_STORE)
        self.reasoning_agent = ComparisonReasoningAgent(LLM)
        self.validation_agent = ValidationAgent()

    def _classify_intent(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["mandatory", "requirement", "satisf", "compliant", "meets"]):
            return "requirements_check"
        if any(k in q for k in ["price", "pricing", "cost", "annual", "cheaper", "budget"]):
            return "pricing_comparison"
        if any(k in q for k in ["clause", "contract", "renewal", "termination", "liability", "sla"]):
            return "contract_analysis"
        if any(k in q for k in ["missing", "gap", "incomplete", "not provided"]):
            return "gap_analysis"
        return "general_qa"

    def handle_query(self, query: str, vendor_filter: Optional[List[str]] = None,
                      top_k: Optional[int] = None) -> ChatResponse:
        agents_used = ["PlannerAgent"]
        intent = self._classify_intent(query)
        agents_used.append(f"intent={intent}")

        # Requirements-aware retrieval: augment query with requirement context if relevant
        requirements_ctx = None
        if intent in ("requirements_check", "gap_analysis") and REQUIREMENTS_STORE:
            requirements_ctx = REQUIREMENTS_STORE
            agents_used.append("RequirementExtractionAgent(cached)")

        agents_used.append("RetrievalAgent")
        results = self.retrieval_agent.run(query, top_k=top_k, vendor_filter=vendor_filter)

        agents_used.append("ComparisonReasoningAgent")
        answer = self.reasoning_agent.run(query, results, requirements=requirements_ctx)

        agents_used.append("ValidationAgent")
        validation = self.validation_agent.run(answer, results)

        if not validation["passed"] and validation["snippets_available"] > 0:
            answer += ("\n\n⚠️ Validation notice: some statements in this answer could not be "
                       "fully verified against the source documents. Please double-check "
                       "cited sources before relying on this answer.")

        citations = [
            Citation(
                chunk_id=chunk.chunk_id, vendor=chunk.vendor, doc_type=chunk.doc_type,
                filename=chunk.filename, snippet=chunk.text[:300], score=round(score, 3),
            )
            for chunk, score in results
        ]

        return ChatResponse(
            answer=answer, citations=citations, agents_used=agents_used,
            validation=validation, intent=intent,
        )

ORCHESTRATOR = Orchestrator()

# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="Procurement Vendor Comparison — Multi-Agent RAG System",
    description="Upload vendor proposals, pricing, requirements, and contracts; "
                "then query them via a grounded, multi-agent chatbot.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "llm_configured": bool(settings.GROQ_API_KEY),
        "embedding_model": settings.EMBEDDING_MODEL,
        "documents_indexed": len(VECTOR_STORE.chunks),
        "requirements_loaded": len(REQUIREMENTS_STORE),
    }

@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    vendor: str = Form(...),
    doc_type: DocType = Form(...),
):
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_MB}MB limit")

    ext = (file.filename or "").lower().split(".")[-1]
    doc_id = str(uuid.uuid4())
    requirements_count = None

    try:
        if ext == "pdf":
            text = normalize_text(load_pdf(raw))
        elif ext == "txt":
            text = normalize_text(load_txt(raw))
        elif ext == "csv":
            text, df = load_csv(raw)
            text = normalize_text(text)
        elif ext in ("xlsx", "xls"):
            text, df = load_excel(raw)
            text = normalize_text(text)
            if doc_type == DocType.REQUIREMENTS:
                new_reqs = RequirementExtractionAgent().run(df, source_doc=file.filename)
                REQUIREMENTS_STORE.extend(new_reqs)
                requirements_count = len(new_reqs)
        else:
            raise HTTPException(400, f"Unsupported file extension: .{ext}")
    except DocumentProcessingError as e:
        raise HTTPException(422, str(e))

    if not text.strip():
        raise HTTPException(422, "No extractable text found in document")

    pieces = chunk_text(text)
    chunks = [
        Chunk(
            chunk_id=hashlib.sha1(f"{doc_id}-{i}".encode()).hexdigest()[:12],
            doc_id=doc_id, vendor=vendor, doc_type=doc_type,
            filename=file.filename or "unknown", text=piece, position=i,
        )
        for i, piece in enumerate(pieces)
    ]
    VECTOR_STORE.add(chunks)

    DOCUMENT_REGISTRY[doc_id] = {
        "vendor": vendor, "doc_type": doc_type.value, "filename": file.filename,
        "chunks": len(chunks), "uploaded_at": datetime.utcnow().isoformat(),
    }

    logger.info(f"Uploaded {file.filename} ({doc_type.value}, vendor={vendor}) -> {len(chunks)} chunks")

    return UploadResponse(
        doc_id=doc_id, vendor=vendor, doc_type=doc_type, filename=file.filename or "unknown",
        chunks_created=len(chunks), requirements_extracted=requirements_count,
    )

@app.get("/vendors")
def list_vendors():
    return {
        "vendors": VECTOR_STORE.vendors(),
        "documents_by_vendor": VECTOR_STORE.doc_count_by_vendor(),
        "documents": DOCUMENT_REGISTRY,
    }

@app.get("/requirements")
def list_requirements():
    return {"count": len(REQUIREMENTS_STORE), "requirements": [r.dict() for r in REQUIREMENTS_STORE]}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(400, "query must not be empty")
    try:
        return ORCHESTRATOR.handle_query(req.query, vendor_filter=req.vendor_filter, top_k=req.top_k)
    except DocumentProcessingError as e:
        raise HTTPException(500, str(e))

@app.delete("/reset")
def reset_store():
    """Clears all in-memory state — useful for demos/tests."""
    global VECTOR_STORE
    VECTOR_STORE = VectorStore()
    ORCHESTRATOR.retrieval_agent.store = VECTOR_STORE
    REQUIREMENTS_STORE.clear()
    DOCUMENT_REGISTRY.clear()
    return {"status": "reset"}

# ---- Minimal built-in chat UI (swap for your real frontend) ----------------

CHAT_UI_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Procurement AI Assistant</title>
  <style>
    body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
    #chat { border: 1px solid #ddd; border-radius: 8px; height: 420px; overflow-y: auto; padding: 16px; margin-bottom: 12px; background: #fafafa; }
    .msg { margin-bottom: 14px; }
    .user { color: #1a4d8f; font-weight: 600; }
    .bot { white-space: pre-wrap; }
    .cite { font-size: 12px; color: #666; margin-top: 6px; }
    input, button { padding: 10px; font-size: 14px; }
    input { width: 70%; }
    button { cursor: pointer; }
    .upload { margin-bottom: 20px; padding: 12px; background: #f0f4ff; border-radius: 8px; }
  </style>
</head>
<body>
  <h2>📋 Procurement Vendor Comparison Assistant</h2>
  <div class="upload">
    <b>Upload document:</b><br><br>
    <input type="file" id="file"/>
    <input type="text" id="vendor" placeholder="Vendor name" />
    <select id="doctype">
      <option value="proposal">Proposal (PDF)</option>
      <option value="pricing">Pricing (CSV)</option>
      <option value="requirements">Requirements (Excel)</option>
      <option value="contract">Contract (TXT/PDF)</option>
    </select>
    <button onclick="upload()">Upload</button>
    <div id="uploadStatus"></div>
  </div>
  <div id="chat"></div>
  <input id="query" placeholder="Ask about vendors, pricing, requirements, clauses..." onkeydown="if(event.key==='Enter') send()"/>
  <button onclick="send()">Send</button>

  <script>
    async function upload() {
      const f = document.getElementById('file').files[0];
      const vendor = document.getElementById('vendor').value;
      const doctype = document.getElementById('doctype').value;
      if (!f || !vendor) { alert('Select a file and enter vendor name'); return; }
      const fd = new FormData();
      fd.append('file', f); fd.append('vendor', vendor); fd.append('doc_type', doctype);
      const res = await fetch('/documents/upload', { method: 'POST', body: fd });
      const data = await res.json();
      document.getElementById('uploadStatus').innerText = res.ok
        ? `✓ Uploaded ${data.filename}: ${data.chunks_created} chunks indexed` + (data.requirements_extracted ? `, ${data.requirements_extracted} requirements extracted` : '')
        : `✗ ${data.detail}`;
    }

    async function send() {
      const qEl = document.getElementById('query');
      const query = qEl.value.trim();
      if (!query) return;
      const chat = document.getElementById('chat');
      chat.innerHTML += `<div class="msg user">You: ${query}</div>`;
      qEl.value = '';
      const res = await fetch('/chat', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ query })
      });
      const data = await res.json();
      const cites = (data.citations || []).map((c,i) => `[S${i+1}] ${c.vendor}/${c.doc_type}/${c.filename}`).join('<br>');
      chat.innerHTML += `<div class="msg bot">Assistant (${data.validation.confidence} confidence): ${data.answer}
        <div class="cite">${cites}</div></div>`;
      chat.scrollTop = chat.scrollHeight;
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return CHAT_UI_HTML