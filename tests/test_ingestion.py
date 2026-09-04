from src.ingestion.chunker import chunk_sections
from src.ingestion.parsers import ParsedSection, parse_document


def test_text_parser_preserves_source_location() -> None:
    sections = parse_document("notes.txt", b"annual pricing")
    assert sections[0].location == "Text"
    assert sections[0].filename == "notes.txt"


def test_chunker_overlaps_words_and_preserves_metadata() -> None:
    section = ParsedSection("proposal.txt", "one two three four five", "Page 2")
    chunks = chunk_sections([section], chunk_tokens=3, overlap_tokens=1)
    assert [chunk.text for chunk in chunks] == ["one two three", "three four five"]
    assert chunks[1].metadata["location"] == "Page 2"