"""Environment-backed application settings."""

from functools import lru_cache
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


class Settings(BaseModel):
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-20b", alias="GROQ_MODEL")
    chromadb_host: str = Field(default="localhost", alias="CHROMADB_HOST")
    chromadb_port: int = Field(default=8000, alias="CHROMADB_PORT")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME"
    )
    chroma_collection: str = Field(default="procurement_docs", alias="CHROMA_COLLECTION")
    retrieval_top_k: int = Field(default=6, alias="RETRIEVAL_TOP_K", ge=1, le=20)
    relevance_threshold: float = Field(default=0.35, alias="RELEVANCE_THRESHOLD", ge=0, le=1)
    max_upload_mb: int = Field(default=25, alias="MAX_UPLOAD_MB", ge=1)

    model_config = {"populate_by_name": True}


@lru_cache
def get_settings() -> Settings:
    values = {key: value for key, value in os.environ.items() if key.isupper()}
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application configuration: {exc}") from exc