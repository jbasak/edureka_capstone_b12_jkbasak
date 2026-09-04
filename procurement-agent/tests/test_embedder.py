"""
tests/test_embedder.py

Unit tests for app/ingestion/embedder.py

The SentenceTransformer model is patched so tests:
  - Run without downloading ~90 MB of model weights
  - Execute in milliseconds
  - Are fully deterministic

Each test patches _load_model() to return a mock whose .encode() returns
numpy-style arrays of the expected shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

VECTOR_DIM = 384


def _make_mock_model(dim: int = VECTOR_DIM):
    """Return a mock SentenceTransformer that returns deterministic embeddings."""
    model = MagicMock()

    def encode(texts, **kwargs):
        # Return a (len(texts), dim) numpy array of 1.0 values
        arr = np.ones((len(texts), dim), dtype=np.float32)
        # Make each vector slightly unique by using its index
        for i in range(len(texts)):
            arr[i, 0] = float(i + 1)
        return arr

    model.encode.side_effect = encode
    model.get_sentence_embedding_dimension.return_value = dim
    return model


# ── embed_texts ───────────────────────────────────────────────────────────────

class TestEmbedTexts:
    def test_correct_number_of_vectors(self):
        from app.ingestion.embedder import embed_texts

        with patch("app.ingestion.embedder._load_model", return_value=_make_mock_model()):
            result = embed_texts(["text one", "text two", "text three"])

        assert len(result) == 3

    def test_correct_vector_dimension(self):
        from app.ingestion.embedder import embed_texts

        with patch("app.ingestion.embedder._load_model", return_value=_make_mock_model()):
            result = embed_texts(["hello world"])

        assert len(result[0]) == VECTOR_DIM

    def test_returns_list_of_lists(self):
        from app.ingestion.embedder import embed_texts

        with patch("app.ingestion.embedder._load_model", return_value=_make_mock_model()):
            result = embed_texts(["foo"])

        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    def test_empty_input_returns_empty_list(self):
        from app.ingestion.embedder import embed_texts

        with patch("app.ingestion.embedder._load_model", return_value=_make_mock_model()):
            result = embed_texts([])

        assert result == []

    def test_vectors_are_distinct_for_different_inputs(self):
        from app.ingestion.embedder import embed_texts

        with patch("app.ingestion.embedder._load_model", return_value=_make_mock_model()):
            result = embed_texts(["first sentence", "second sentence"])

        # Our mock encodes index into position 0, so vectors differ
        assert result[0][0] != result[1][0]

    def test_batch_of_100_texts(self):
        from app.ingestion.embedder import embed_texts

        texts = [f"sentence number {i}" for i in range(100)]
        with patch("app.ingestion.embedder._load_model", return_value=_make_mock_model()):
            result = embed_texts(texts)

        assert len(result) == 100
        for vec in result:
            assert len(vec) == VECTOR_DIM

    def test_raises_for_non_list_input(self):
        from app.ingestion.embedder import embed_texts

        with pytest.raises(ValueError, match="expects a list"):
            embed_texts("not a list")  # type: ignore[arg-type]


# ── embed_query ───────────────────────────────────────────────────────────────

class TestEmbedQuery:
    def test_returns_single_vector(self):
        from app.ingestion.embedder import embed_query

        with patch("app.ingestion.embedder._load_model", return_value=_make_mock_model()):
            result = embed_query("Which vendor offers the lowest price?")

        assert isinstance(result, list)
        assert len(result) == VECTOR_DIM

    def test_all_floats(self):
        from app.ingestion.embedder import embed_query

        with patch("app.ingestion.embedder._load_model", return_value=_make_mock_model()):
            result = embed_query("test query")

        assert all(isinstance(v, float) for v in result)

    def test_empty_query_raises(self):
        from app.ingestion.embedder import embed_query

        with pytest.raises(ValueError, match="must not be empty"):
            embed_query("")

    def test_whitespace_only_query_raises(self):
        from app.ingestion.embedder import embed_query

        with pytest.raises(ValueError, match="must not be empty"):
            embed_query("   ")

    def test_same_query_produces_same_vector(self):
        """Consistency: same input text → same output (mock is deterministic)."""
        from app.ingestion.embedder import embed_query

        model = _make_mock_model()
        with patch("app.ingestion.embedder._load_model", return_value=model):
            v1 = embed_query("cloud pricing comparison")
            v2 = embed_query("cloud pricing comparison")

        assert v1 == v2
