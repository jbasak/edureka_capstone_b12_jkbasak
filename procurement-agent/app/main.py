"""
app/main.py
FastAPI application entry point.
Registers routers for /documents, /chat, and /health.
Initialises the Qdrant collection on startup.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import documents, chat
from app.vector_store.qdrant_client import get_qdrant_manager

settings = get_settings()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: startup / shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise resources on startup; clean up on shutdown."""
    logger.info("Starting Procurement Agent API …")
    qdrant_mgr = get_qdrant_manager()
    qdrant_mgr.ensure_collection()
    logger.info(
        "Qdrant collection '%s' ready at %s:%d",
        settings.qdrant_collection,
        settings.qdrant_host,
        settings.qdrant_port,
    )
    yield
    logger.info("Shutting down Procurement Agent API.")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Procurement & Vendor Evaluation Assistant",
    description=(
        "AI-powered multi-agent system for RFP analysis, vendor comparison, "
        "and compliance validation. Upload documents via /documents and ask "
        "natural-language questions via /chat."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (useful for a local web UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents.router, prefix="/documents", tags=["Ingestion"])
app.include_router(chat.router, prefix="/chat", tags=["Query"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    """
    Returns service health, Qdrant connectivity, and current collection stats.
    """
    qdrant_mgr = get_qdrant_manager()
    try:
        info = qdrant_mgr.collection_info()
        qdrant_status = "connected"
        vector_count = info.vectors_count if info else 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant health check failed: %s", exc)
        qdrant_status = "unavailable"
        vector_count = None

    return {
        "status": "healthy",
        "qdrant": qdrant_status,
        "collection": settings.qdrant_collection,
        "vector_count": vector_count,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.groq_model,
    }


@app.get("/", tags=["Health"])
def root():
    """Redirect hint — use /docs for the interactive API explorer."""
    return {
        "message": "Procurement Agent API is running. Visit /docs for the API explorer."
    }
