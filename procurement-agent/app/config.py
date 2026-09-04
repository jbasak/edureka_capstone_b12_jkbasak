"""
app/config.py
Application configuration loaded from .env via pydantic-settings.
Single Settings instance shared across all modules.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Groq LLM ────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # ── Embedding model (HuggingFace / sentence-transformers) ────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dimension: int = 384  # all-MiniLM-L6-v2 output size

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 900        # tokens per chunk
    chunk_overlap: int = 150     # overlap tokens (~17%)

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = 6

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    # persist_dir: local directory where ChromaDB stores its data files.
    # Use ":memory:" in tests (handled directly in conftest via EphemeralClient).
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "procurement_docs"

    # ── Upload limits ────────────────────────────────────────────────────────
    max_upload_mb: int = 25

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── App ───────────────────────────────────────────────────────────────────
    app_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
