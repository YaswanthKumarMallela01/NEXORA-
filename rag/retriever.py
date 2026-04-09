"""
NEXORA — RAG Retriever
Pinecone-backed similarity search with metadata filtering for
CoachAgent (resource recommendations) and ResumeAgent (JD matching).
"""

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

from config import get_settings

logger = logging.getLogger("nexora.rag.retriever")

_embeddings: Optional[HuggingFaceEmbeddings] = None
_pinecone_index = None


# ────────────────────────────────────────────────────────────
#  Initialization
# ────────────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Cached HuggingFace embeddings model."""
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Embeddings model loaded: {settings.EMBEDDING_MODEL}")
    return _embeddings


def _get_index():
    """Cached Pinecone index connection."""
    global _pinecone_index
    if _pinecone_index is None:
        settings = get_settings()
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _pinecone_index = pc.Index(
            settings.PINECONE_INDEX_NAME,
            host=settings.PINECONE_HOST,
        )
        logger.info(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
    return _pinecone_index


# ────────────────────────────────────────────────────────────
#  Core Retrieval
# ────────────────────────────────────────────────────────────

def query_similar(
    query: str,
    category: Optional[str] = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Perform similarity search on Pinecone.
    
    Args:
        query: Natural language query string
        category: Optional filter — one of:
            'job_descriptions', 'dsa', 'hr_guides',
            'company_profiles', 'learning_resources', 'general'
        top_k: Number of results to return
    
    Returns:
        List of dicts with keys: text, score, metadata
    """
    settings = get_settings()
    top_k = top_k or settings.RAG_TOP_K

    embeddings = _get_embeddings()
    index = _get_index()

    # Embed the query
    query_vector = embeddings.embed_query(query)

    # Build filter
    filter_dict = {}
    if category:
        filter_dict["category"] = {"$eq": category}

    # Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None,
    )

    # Parse results
    documents = []
    for match in results.get("matches", []):
        doc = {
            "text": match.get("metadata", {}).get("text", ""),
            "score": match.get("score", 0.0),
            "metadata": {
                k: v for k, v in match.get("metadata", {}).items() if k != "text"
            },
        }
        documents.append(doc)

    logger.info(f"Retrieved {len(documents)} docs for query (category={category})")
    return documents


# ────────────────────────────────────────────────────────────
#  Specialized Retrievers
# ────────────────────────────────────────────────────────────

def get_job_descriptions(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve top matching job descriptions for resume analysis."""
    return query_similar(query, category="job_descriptions", top_k=top_k)


def get_learning_resources(skill: str, top_k: int = 5) -> list[dict]:
    """Retrieve learning resources for a specific skill."""
    return query_similar(
        f"learning resources tutorials courses for {skill}",
        category="learning_resources",
        top_k=top_k,
    )


def get_dsa_resources(topic: str, top_k: int = 5) -> list[dict]:
    """Retrieve DSA study materials for a specific topic."""
    return query_similar(
        f"DSA data structures algorithms {topic}",
        category="dsa",
        top_k=top_k,
    )


def get_hr_guides(topic: str, top_k: int = 3) -> list[dict]:
    """Retrieve HR interview preparation guides."""
    return query_similar(
        f"HR interview behavioral questions {topic}",
        category="hr_guides",
        top_k=top_k,
    )


def get_company_profiles(company: str, top_k: int = 3) -> list[dict]:
    """Retrieve company profiles and hiring information."""
    return query_similar(
        f"company profile hiring process {company}",
        category="company_profiles",
        top_k=top_k,
    )


# ────────────────────────────────────────────────────────────
#  Context Builder (for Agent prompts)
# ────────────────────────────────────────────────────────────

def build_context_string(docs: list[dict], max_chars: int = 3000) -> str:
    """
    Build a formatted context string from retrieved documents
    for injection into agent prompts.
    """
    if not docs:
        return "No relevant documents found."

    context_parts = []
    char_count = 0

    for i, doc in enumerate(docs, 1):
        text = doc["text"]
        source = doc["metadata"].get("source", "unknown")
        score = doc["score"]

        entry = f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}\n"

        if char_count + len(entry) > max_chars:
            break

        context_parts.append(entry)
        char_count += len(entry)

    return "\n---\n".join(context_parts)
