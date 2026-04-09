"""
NEXORA — LLM Orchestrator
Multi-LLM routing with automatic fallback.

Routing strategy:
  • Groq (llama-3.1-70b) — primary reasoning, fast responses
  • Gemini 1.5 Flash — interview agent (large context window)
  • Together AI (Mixtral-8x7B) — fallback when Groq is rate-limited
  • HuggingFace (all-MiniLM-L6-v2) — embeddings only (via rag/)
"""

import logging
from typing import Optional

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableWithFallbacks

from config import get_settings

logger = logging.getLogger("nexora.chains.orchestrator")


# ────────────────────────────────────────────────────────────
#  LLM Factory Functions
# ────────────────────────────────────────────────────────────

def get_groq_llm(
    temperature: float = 0.3,
    max_tokens: int = 4096,
    model: Optional[str] = None,
) -> ChatGroq:
    """
    Primary LLM — Groq (Llama 3.1 70B Versatile).
    Ultra-fast inference for reasoning tasks.
    """
    settings = get_settings()
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=model or settings.GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_gemini_llm(
    temperature: float = 0.4,
    max_tokens: int = 8192,
    model: Optional[str] = None,
) -> ChatGoogleGenerativeAI:
    """
    Interview LLM — Google Gemini 1.5 Flash.
    Large context window for tracking full interview sessions.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        google_api_key=settings.GOOGLE_API_KEY,
        model=model or settings.GEMINI_MODEL,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def get_together_llm(
    temperature: float = 0.3,
    max_tokens: int = 4096,
    model: Optional[str] = None,
) -> BaseChatModel:
    """
    Fallback LLM — Together AI (Mixtral 8x7B).
    Activated when Groq is rate-limited.
    """
    from langchain_community.chat_models import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
        model=model or settings.TOGETHER_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ────────────────────────────────────────────────────────────
#  LLM with Automatic Fallback
# ────────────────────────────────────────────────────────────

def get_llm_with_fallback(
    primary: str = "groq",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> RunnableWithFallbacks:
    """
    Returns an LLM with automatic fallback chain.
    
    If primary (Groq) fails due to rate limiting or errors,
    automatically falls back to Together AI (Mixtral).
    """
    if primary == "groq":
        primary_llm = get_groq_llm(temperature=temperature, max_tokens=max_tokens)
        fallback_llm = get_together_llm(temperature=temperature, max_tokens=max_tokens)
    elif primary == "gemini":
        primary_llm = get_gemini_llm(temperature=temperature, max_tokens=max_tokens)
        fallback_llm = get_groq_llm(temperature=temperature, max_tokens=max_tokens)
    else:
        primary_llm = get_groq_llm(temperature=temperature, max_tokens=max_tokens)
        fallback_llm = get_together_llm(temperature=temperature, max_tokens=max_tokens)

    logger.info(f"LLM initialized: primary={primary}, fallback=together")
    return primary_llm.with_fallbacks([fallback_llm])


# ────────────────────────────────────────────────────────────
#  Agent Router
# ────────────────────────────────────────────────────────────

AGENT_LLM_MAP = {
    "resume": "groq",
    "coach": "groq",
    "interview": "gemini",
    "alert": "groq",
}


def get_agent_llm(agent_type: str, **kwargs) -> BaseChatModel:
    """
    Get the appropriate LLM for a specific agent type.
    
    Args:
        agent_type: One of 'resume', 'coach', 'interview', 'alert'
    
    Returns:
        Configured LLM (with fallback for groq-based agents)
    """
    primary = AGENT_LLM_MAP.get(agent_type, "groq")

    if primary == "gemini":
        # Gemini for interview — large context, no fallback needed
        return get_gemini_llm(**kwargs)
    else:
        # Groq with Together AI fallback
        return get_llm_with_fallback(primary=primary, **kwargs)
