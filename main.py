"""
NEXORA — Main Application Entry Point
FastAPI server with multi-agent AI, RAG pipeline, Supabase integration,
and Stitch-generated Neural Minimalism frontend.

Run locally:  .venv\\Scripts\\python.exe -m uvicorn main:app --reload --port 8000
Deploy:        Render.com (see render.yaml / Procfile)
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

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
    system_router,
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

# Frontend directory path
FRONTEND_DIR = Path(__file__).parent / "frontend"


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

    # Pre-load Embeddings (Prevents timeout on first RAG query)
    try:
        from rag.retriever import _get_embeddings
        logger.info("📡 Pre-loading Embeddings model...")
        _get_embeddings()
        logger.info("✓ Embeddings model pre-loaded")
    except Exception as e:
        logger.warning(f"⚠ Embeddings pre-load failed (will load on demand): {e}")

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

    logger.info(f"✓ Frontend UI: {FRONTEND_DIR} ({len(list(FRONTEND_DIR.glob('*.html')))} pages)")
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

# ── API Routers ──
app.include_router(auth_router)
app.include_router(resume_router)
app.include_router(coach_router)
app.include_router(interview_router)
app.include_router(dashboard_router)
app.include_router(alerts_router)
app.include_router(tasks_router)
app.include_router(system_router)

# ── Static files (frontend assets if any) ──
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ────────────────────────────────────────────────────────────
#  Frontend Page Routes (Stitch Neural Minimalism UI)
# ────────────────────────────────────────────────────────────

def _serve_page(filename: str) -> HTMLResponse:
    """Read and serve an HTML file from the frontend directory."""
    filepath = FRONTEND_DIR / filename
    if filepath.exists():
        return HTMLResponse(content=filepath.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Page not found</h1>", status_code=404)


@app.get("/", tags=["Frontend"], include_in_schema=False)
async def root():
    """Redirect root to login/landing page."""
    return RedirectResponse(url="/login")


@app.get("/login", tags=["Frontend"], include_in_schema=False)
async def login_page():
    """Landing & Login page — Neural Minimalism design."""
    return _serve_page("login.html")


@app.get("/auth/callback", tags=["Frontend"], include_in_schema=False)
async def auth_callback_page():
    """Supabase email-confirmation redirect — stores tokens from URL hash."""
    return _serve_page("auth-callback.html")


@app.get("/student", tags=["Frontend"], include_in_schema=False)
async def student_dashboard_page():
    """Student Dashboard — readiness score, tasks, skills, alerts."""
    return _serve_page("dashboard.html")


@app.get("/resume", tags=["Frontend"], include_in_schema=False)
async def resume_scan_page():
    """Resume upload — PDF analysis via ResumeAgent."""
    return _serve_page("resume.html")


@app.get("/coach", tags=["Frontend"], include_in_schema=False)
async def coach_page():
    """AI Coach — stateful chat interface with session insights."""
    return _serve_page("coach.html")


@app.get("/interview", tags=["Frontend"], include_in_schema=False)
async def interview_page():
    """Mock Interview — interactive AI interview simulator."""
    return _serve_page("interview.html")


@app.get("/roadmap", tags=["Frontend"], include_in_schema=False)
async def roadmap_page():
    """Career Roadmap — weekly prep plan with milestones."""
    return _serve_page("roadmap.html")


@app.get("/tpc", tags=["Frontend"], include_in_schema=False)
async def tpc_dashboard_page():
    """TPC Admin Dashboard — all students, risk tracking."""
    return _serve_page("tpc.html")


@app.get("/auth-callback", tags=["Frontend"], include_in_schema=False)
async def auth_callback_page():
    """Supabase email confirmation callback handler."""
    return _serve_page("auth-callback.html")


# ────────────────────────────────────────────────────────────
#  Health & System Endpoints
# ────────────────────────────────────────────────────────────

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


@app.get("/api/info", tags=["System"])
async def api_info():
    """API info endpoint — returns system status."""
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
        "frontend_pages": [
            "/login — Landing & Authentication",
            "/auth/callback — Email confirmation redirect (Supabase)",
            "/student — Student Dashboard",
            "/resume — Resume PDF scan & analysis",
            "/coach — AI Career Coach Chat",
            "/interview — Mock Interview Simulator",
            "/roadmap — Career Roadmap",
            "/tpc — TPC Admin Dashboard",
        ],
        "diagnostics": "/api/system/status",
    }


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
