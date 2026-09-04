"""Approximate token chunking with overlap and source metadata."""

from dataclasses import dataclass

from .parsers import ParsedSection


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    metadata: dict[str, str]


def chunk_sections(
    sections: list[ParsedSection], chunk_tokens: int = 500, overlap_tokens: int = 50
) -> list[DocumentChunk]:
    if chunk_tokens <= overlap_tokens or overlap_tokens < 0:
        raise ValueError("chunk_tokens must be greater than a non-negative overlap_tokens")

    chunks: list[DocumentChunk] = []
    step = chunk_tokens - overlap_tokens
    for section in sections:
        words = section.text.split()
        for start in range(0, len(words), step):
            text = " ".join(words[start : start + chunk_tokens]).strip()
            if text:
                chunks.append(
                    DocumentChunk(
                        text=text,
                        metadata={"filename": section.filename, "location": section.location},
                    )
                )
            if start + chunk_tokens >= len(words):
                break
    return chunks