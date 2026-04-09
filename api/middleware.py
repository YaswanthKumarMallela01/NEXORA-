"""
NEXORA — API Middleware
JWT authentication, rate limiting, CORS, and error handling.
"""

import time
import logging
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_settings

logger = logging.getLogger("nexora.api.middleware")


# ────────────────────────────────────────────────────────────
#  Rate Limiting
# ────────────────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.
    Limits requests per IP per time window.
    """

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/", "/docs", "/openapi.json"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        self.requests[client_ip] = [
            t for t in self.requests[client_ip]
            if now - t < self.window_seconds
        ]

        # Check limit
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
            )

        self.requests[client_ip].append(now)
        return await call_next(request)


# ────────────────────────────────────────────────────────────
#  Request Logging
# ────────────────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests with timing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} "
            f"({duration:.3f}s)"
        )

        # Add timing header
        response.headers["X-Process-Time"] = f"{duration:.3f}"
        return response


# ────────────────────────────────────────────────────────────
#  CORS Configuration
# ────────────────────────────────────────────────────────────

def get_cors_origins() -> list[str]:
    """Get allowed CORS origins based on environment."""
    settings = get_settings()

    origins = [
        settings.NEXT_PUBLIC_APP_URL,
        "https://nexora.vercel.app",
    ]

    # Add localhost for development
    if settings.ENVIRONMENT != "production":
        origins.extend([
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ])

    return origins


def setup_cors(app):
    """Add CORS middleware to the FastAPI app."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time"],
    )


# ────────────────────────────────────────────────────────────
#  n8n Webhook Validation
# ────────────────────────────────────────────────────────────

async def validate_webhook_key(request: Request):
    """Validate the n8n webhook API key."""
    settings = get_settings()

    if not settings.N8N_API_KEY:
        return  # Skip if no key configured

    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

    if api_key != settings.N8N_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid webhook API key")
