"""FastAPI entry point for procurement document ingestion and grounded queries."""

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.agents.execution_agent import execute_query
from src.agents.requirement_agent import evaluate_requirements
from src.agents.retrieval_agent import retrieve
from src.config import get_settings
from src.ingestion.chunker import chunk_sections
from src.ingestion.parsers import parse_document
from src.vectorstore.chroma_client import add_chunks

app = FastAPI(title="Procurement and Vendor Evaluation Assistant", version="1.0.0")


class QueryRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)

    def normalized_prompt(self) -> str:
        return self.prompt.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
    settings = get_settings()
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")
    try:
        sections = parse_document(file.filename or "upload", content)
        chunks = chunk_sections(sections)
        if not chunks:
            raise ValueError("Document contains no extractable text")
        count = add_chunks(chunks)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"filename": file.filename, "chunks_stored": count, "status": "indexed"}


@app.post("/requirements", status_code=status.HTTP_200_OK)
async def evaluate_compliance(file: UploadFile = File(...)) -> dict[str, object]:
    content = await file.read()
    try:
        matrix = evaluate_requirements(content, file.filename or "requirements.xlsx")
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"filename": file.filename, "matrix": matrix}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    prompt = request.normalized_prompt()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must contain non-whitespace text")
    try:
        answer, chunks = execute_query(prompt)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="LLM provider request failed") from exc
    return QueryResponse(answer=answer, sources=list(dict.fromkeys(str(chunk.metadata) for chunk in chunks)))