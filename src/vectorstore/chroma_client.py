"""ChromaDB access with a local HuggingFace embedding function."""

from functools import lru_cache
import hashlib
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from src.config import get_settings


class SentenceTransformerEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(list(input), normalize_embeddings=True).tolist()


@lru_cache
def get_embedding_function() -> SentenceTransformerEmbedding:
    return SentenceTransformerEmbedding(get_settings().embedding_model_name)


@lru_cache
def get_collection() -> Any:
    settings = get_settings()
    client = chromadb.HttpClient(host=settings.chromadb_host, port=settings.chromadb_port)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Any]) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    collection.upsert(
        ids=[
            hashlib.sha256(
                f"{chunk.metadata['filename']}:{chunk.metadata['location']}:{chunk.text}".encode()
            ).hexdigest()
            for chunk in chunks
        ],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )
    return len(chunks)


def query_chunks(query: str, top_k: int) -> list[dict[str, Any]]:
    result = get_collection().query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {"text": text, "metadata": metadata or {}, "distance": float(distance)}
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]