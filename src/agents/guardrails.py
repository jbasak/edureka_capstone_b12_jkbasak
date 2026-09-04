"""Grounding and citation guardrails for generated responses."""

from collections.abc import Iterable
import re

from .retrieval_agent import RetrievedChunk

FALLBACK_MESSAGE = "Information not found in context"


def citation(metadata: dict[str, str]) -> str:
    filename = metadata.get("filename", "Unknown source")
    location = metadata.get("location", "Unknown location")
    return f"[{filename}, {location}]"


def grounded_context(chunks: Iterable[RetrievedChunk]) -> str:
    return "\n\n".join(f"{chunk.text} {citation(chunk.metadata)}" for chunk in chunks)


def enforce_grounding(response: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return FALLBACK_MESSAGE
    sources = list(dict.fromkeys(citation(chunk.metadata) for chunk in chunks))
    fallback = " ".join(sources)
    lines = [line.rstrip() for line in response.splitlines() if line.strip()]
    if not lines:
        return FALLBACK_MESSAGE
    return "\n".join(line if re.search(r"\[[^\]]+, [^\]]+\]", line) else f"{line} {fallback}" for line in lines)