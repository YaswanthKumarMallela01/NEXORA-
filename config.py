"""
NEXORA — Central Configuration
Loads and validates all environment variables via pydantic-settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """All NEXORA configuration loaded from .env"""

    # ── LLM APIs ────────────────────────────────────────
    GROQ_API_KEY: str = Field(..., description="Groq API key for Llama 3.1")
    GOOGLE_API_KEY: str = Field(..., description="Google Gemini API key")
    HUGGINGFACE_API_KEY: str = Field(..., description="HuggingFace API key")
    TOGETHER_API_KEY: str = Field(..., description="Together AI API key for fallback")

    # ── Vector Store ────────────────────────────────────
    PINECONE_API_KEY: str = Field(..., description="Pinecone API key")
    PINECONE_HOST: str = Field(..., description="Pinecone index host URL")
    PINECONE_INDEX_NAME: str = Field(default="nexora-rag")

    # ── Supabase ────────────────────────────────────────
    SUPABASE_URL: str = Field(..., description="Supabase project URL")
    SUPABASE_ANON_KEY: str = Field(..., description="Supabase anon/public key")
    SUPABASE_SERVICE_ROLE_KEY: str = Field(..., description="Supabase service role key")

    # ── Email ───────────────────────────────────────────
    RESEND_API_KEY: str = Field(..., description="Resend API key")
    FROM_EMAIL: str = Field(default="onboarding@resend.dev")

    # ── n8n Automation ──────────────────────────────────
    N8N_WEBHOOK_URL: str = Field(default="")
    N8N_API_KEY: str = Field(default="")

    # ── App Config ──────────────────────────────────────
    NEXT_PUBLIC_APP_URL: str = Field(default="https://nexora.vercel.app")
    # Where Supabase redirects after email confirmation (must be listed in Supabase Dashboard → Auth → URL Configuration → Redirect URLs)
    AUTH_EMAIL_REDIRECT_URL: str = Field(
        default="",
        description="Full URL to /auth/callback; if empty, derived from NEXT_PUBLIC_APP_URL",
    )
    JWT_SECRET: str = Field(..., description="JWT signing secret")
    ENVIRONMENT: str = Field(default="development")

    # ── Model Defaults ──────────────────────────────────
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    GEMINI_MODEL: str = Field(default="gemini-3.1-flash-lite-preview")
    TOGETHER_MODEL: str = Field(default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")

    # ── RAG Config ──────────────────────────────────────
    RAG_CHUNK_SIZE: int = Field(default=500)
    RAG_CHUNK_OVERLAP: int = Field(default=50)
    RAG_TOP_K: int = Field(default=5)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — loaded once, reused everywhere."""
    return Settings()
