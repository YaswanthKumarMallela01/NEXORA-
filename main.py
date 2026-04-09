"""
NEXORA — Main Application Entry Point
FastAPI server with multi-agent AI, RAG pipeline, and Supabase integration.

Run locally:  uvicorn main:app --reload --port 8000
Deploy:        Vercel serverless (see vercel.json)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import get_settings
from api.middleware import setup_cors, RateLimitMiddleware, RequestLoggingMiddleware
from api.routes import (
    auth_router,
    resume_router,
    coach_router,
    interview_router,
    dashboard_router,
    alerts_router,
    tasks_router,
)

# ────────────────────────────────────────────────────────────
#  Logging Setup
# ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nexora")


# ────────────────────────────────────────────────────────────
#  Lifespan (startup / shutdown)
# ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    Startup: validate config, initialize connections.
    Shutdown: cleanup.
    """
    # ── STARTUP ──
    logger.info("=" * 60)
    logger.info("🚀 NEXORA Backend starting up...")
    logger.info("=" * 60)

    settings = get_settings()
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Supabase URL: {settings.SUPABASE_URL}")
    logger.info(f"Pinecone Index: {settings.PINECONE_INDEX_NAME}")

    # Validate critical connections
    try:
        from db.supabase_client import get_supabase
        client = get_supabase()
        logger.info("✓ Supabase client initialized")
    except Exception as e:
        logger.error(f"✗ Supabase connection failed: {e}")

    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        indexes = [idx.name for idx in pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME in indexes:
            logger.info(f"✓ Pinecone index '{settings.PINECONE_INDEX_NAME}' found")
        else:
            logger.warning(f"⚠ Pinecone index '{settings.PINECONE_INDEX_NAME}' not found — run ingestion first")
    except Exception as e:
        logger.error(f"✗ Pinecone connection failed: {e}")

    logger.info("✓ All systems initialized")
    logger.info("=" * 60)

    yield  # ── App runs here ──

    # ── SHUTDOWN ──
    logger.info("NEXORA Backend shutting down...")


# ────────────────────────────────────────────────────────────
#  FastAPI App
# ────────────────────────────────────────────────────────────

app = FastAPI(
    title="NEXORA API",
    description=(
        "Multi-agent AI placement readiness platform.\n\n"
        "**Agents:** ResumeAgent, CoachAgent, InterviewAgent, AlertAgent\n\n"
        "**Powered by:** Groq (Llama 3.1), Gemini 1.5 Flash, "
        "Pinecone RAG, Supabase, LangChain"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware (order matters — first added = outermost) ──
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)
app.add_middleware(RequestLoggingMiddleware)
setup_cors(app)

# ── Routers ──
app.include_router(auth_router)
app.include_router(resume_router)
app.include_router(coach_router)
app.include_router(interview_router)
app.include_router(dashboard_router)
app.include_router(alerts_router)
app.include_router(tasks_router)


# ────────────────────────────────────────────────────────────
#  Root & Health
# ────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """API root — basic info."""
    return {
        "name": "NEXORA",
        "version": "1.0.0",
        "description": "Multi-agent AI placement readiness platform",
        "docs": "/docs",
        "agents": [
            "ResumeAgent — PDF analysis & skill extraction",
            "CoachAgent — Stateful AI mentor with memory",
            "InterviewAgent — Mock interviews with Gemini",
            "AlertAgent — Risk evaluation & email alerts",
        ],
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for monitoring."""
    settings = get_settings()

    status = {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "services": {},
    }

    # Check Supabase
    try:
        from db.supabase_client import get_supabase
        get_supabase()
        status["services"]["supabase"] = "connected"
    except Exception:
        status["services"]["supabase"] = "disconnected"
        status["status"] = "degraded"

    # Check Pinecone
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        indexes = [idx.name for idx in pc.list_indexes()]
        status["services"]["pinecone"] = (
            "connected" if settings.PINECONE_INDEX_NAME in indexes else "index_missing"
        )
    except Exception:
        status["services"]["pinecone"] = "disconnected"
        status["status"] = "degraded"

    return status


# ────────────────────────────────────────────────────────────
#  Global Error Handler
# ────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler to prevent 500s from leaking details."""
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An internal error occurred. Our AI agents are looking into it.",
            "detail": str(exc) if get_settings().ENVIRONMENT != "production" else None,
        },
    )
