"""
app/dependencies.py

Shared FastAPI dependency injectors.

Usage in route handlers
-----------------------
    from app.dependencies import get_qdrant, get_settings_dep

    @router.get("/example")
    def example(qdrant=Depends(get_qdrant)):
        ...

These are thin wrappers so tests can override them via app.dependency_overrides.
"""

from app.config import Settings, get_settings
from app.vector_store.qdrant_client import QdrantManager, get_qdrant_manager


def get_settings_dep() -> Settings:
    """FastAPI dependency that returns the cached Settings singleton."""
    return get_settings()


def get_qdrant() -> QdrantManager:
    """FastAPI dependency that returns the cached QdrantManager singleton."""
    return get_qdrant_manager()
