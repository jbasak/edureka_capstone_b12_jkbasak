"""
app/dependencies.py

Shared FastAPI dependency injectors.

Usage in route handlers
-----------------------
    from app.dependencies import get_chroma, get_settings_dep

    @router.get("/example")
    def example(chroma=Depends(get_chroma)):
        ...

These are thin wrappers so tests can override them via app.dependency_overrides.
"""

from app.config import Settings, get_settings
from app.vector_store.chroma_client import ChromaManager, get_chroma_manager


def get_settings_dep() -> Settings:
    """FastAPI dependency that returns the cached Settings singleton."""
    return get_settings()


def get_chroma() -> ChromaManager:
    """FastAPI dependency that returns the cached ChromaManager singleton."""
    return get_chroma_manager()
