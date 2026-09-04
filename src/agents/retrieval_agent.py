"""Retrieve relevant chunks and convert Chroma distances into scores."""

from dataclasses import dataclass

from src.config import get_settings
from src.vectorstore.chroma_client import query_chunks


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, str]
    relevance: float


def retrieve(query: str) -> list[RetrievedChunk]:
    settings = get_settings()
    matches = query_chunks(query, settings.retrieval_top_k)
    return [
        RetrievedChunk(item["text"], item["metadata"], 1 / (1 + item["distance"]))
        for item in matches
        if 1 / (1 + item["distance"]) >= settings.relevance_threshold
    ]