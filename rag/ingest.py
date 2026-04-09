"""
NEXORA — RAG Ingestion Pipeline
Loads documents from placement_resources/, chunks, embeds, and upserts to Pinecone.
Run as: python -m rag.ingest
"""

import os
import logging
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("nexora.rag.ingest")

# Supported file extensions
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json"}


# ────────────────────────────────────────────────────────────
#  Document Loaders
# ────────────────────────────────────────────────────────────

def _detect_category(filepath: str) -> str:
    """Infer document category from filename or parent directory."""
    path_lower = filepath.lower()
    if any(kw in path_lower for kw in ["jd", "job_description", "job-description", "jobs"]):
        return "job_descriptions"
    elif any(kw in path_lower for kw in ["dsa", "algorithm", "data_structure"]):
        return "dsa"
    elif any(kw in path_lower for kw in ["hr", "behavioral", "soft_skill"]):
        return "hr_guides"
    elif any(kw in path_lower for kw in ["company", "profile", "org"]):
        return "company_profiles"
    elif any(kw in path_lower for kw in ["resource", "learn", "tutorial", "course"]):
        return "learning_resources"
    return "general"


def load_documents(source_dir: str) -> list[Document]:
    """Load all supported documents from a directory tree."""
    docs = []
    source_path = Path(source_dir)

    if not source_path.exists():
        logger.warning(f"Source directory not found: {source_dir}")
        return docs

    for filepath in source_path.rglob("*"):
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if filepath.name.startswith("."):
            continue

        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue

            category = _detect_category(str(filepath))
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(filepath.relative_to(source_path)),
                    "category": category,
                    "filename": filepath.name,
                },
            )
            docs.append(doc)
            logger.info(f"Loaded: {filepath.name} (category: {category})")
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


def load_pdf_documents(source_dir: str) -> list[Document]:
    """Load PDF documents using PyMuPDF."""
    import fitz  # PyMuPDF

    docs = []
    source_path = Path(source_dir)

    if not source_path.exists():
        return docs

    for filepath in source_path.rglob("*.pdf"):
        try:
            pdf_doc = fitz.open(str(filepath))
            text = ""
            for page in pdf_doc:
                text += page.get_text()
            pdf_doc.close()

            if not text.strip():
                continue

            category = _detect_category(str(filepath))
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(filepath.relative_to(source_path)),
                    "category": category,
                    "filename": filepath.name,
                },
            )
            docs.append(doc)
            logger.info(f"Loaded PDF: {filepath.name} (category: {category})")
        except Exception as e:
            logger.error(f"Failed to load PDF {filepath}: {e}")

    return docs


# ────────────────────────────────────────────────────────────
#  Chunking
# ────────────────────────────────────────────────────────────

def chunk_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


# ────────────────────────────────────────────────────────────
#  Embedding + Pinecone Upsert
# ────────────────────────────────────────────────────────────

def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Initialize HuggingFace embeddings model."""
    settings = get_settings()
    model = model_name or settings.EMBEDDING_MODEL
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int = 384):
    """Create the Pinecone index if it doesn't exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info(f"Creating Pinecone index '{index_name}' (dim={dimension})")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        logger.info(f"Pinecone index '{index_name}' already exists")


def upsert_to_pinecone(chunks: list[Document], embeddings: HuggingFaceEmbeddings, batch_size: int = 100):
    """Embed chunks and upsert them to Pinecone."""
    settings = get_settings()
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    ensure_pinecone_index(pc, settings.PINECONE_INDEX_NAME)
    index = pc.Index(settings.PINECONE_INDEX_NAME, host=settings.PINECONE_HOST)

    # Process in batches
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        # Embed the batch
        vectors = embeddings.embed_documents(texts)

        # Prepare upsert records
        records = []
        for j, (vec, meta, text) in enumerate(zip(vectors, metadatas, texts)):
            record_id = f"doc_{i + j}"
            meta_with_text = {**meta, "text": text[:1000]}  # Store truncated text in metadata
            records.append({"id": record_id, "values": vec, "metadata": meta_with_text})

        index.upsert(vectors=records)
        logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    logger.info(f"✓ All {total} vectors upserted to Pinecone index '{settings.PINECONE_INDEX_NAME}'")


# ────────────────────────────────────────────────────────────
#  Main Pipeline
# ────────────────────────────────────────────────────────────

def run_ingestion(source_dir: str = "placement_resources"):
    """Full ingestion pipeline: load → chunk → embed → upsert."""
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("NEXORA RAG Ingestion Pipeline")
    logger.info("=" * 60)

    # 1. Load documents
    docs = load_documents(source_dir)
    pdf_docs = load_pdf_documents(source_dir)
    all_docs = docs + pdf_docs

    if not all_docs:
        logger.warning("No documents found. Creating sample data...")
        all_docs = _create_sample_documents()

    # 2. Chunk
    chunks = chunk_documents(
        all_docs,
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
    )

    # 3. Embed + upsert
    embeddings = get_embeddings()
    upsert_to_pinecone(chunks, embeddings)

    logger.info("=" * 60)
    logger.info("✓ Ingestion complete!")
    logger.info("=" * 60)


def _create_sample_documents() -> list[Document]:
    """Create sample documents for testing when no source data exists."""
    samples = [
        Document(
            page_content="""Software Engineer Job Description
            Requirements: Python, JavaScript, React, Node.js, SQL, REST APIs, Git.
            Experience: 0-2 years. Strong problem-solving skills.
            Responsibilities: Design and develop web applications, write clean code,
            participate in code reviews, collaborate with cross-functional teams.
            Nice to have: Docker, Kubernetes, AWS, CI/CD pipelines.""",
            metadata={"source": "sample_jd_1.txt", "category": "job_descriptions", "filename": "sample_jd_1.txt"},
        ),
        Document(
            page_content="""Data Analyst Job Description
            Requirements: Python, SQL, Excel, Tableau/Power BI, Statistics.
            Experience: 0-1 years. Strong analytical and communication skills.
            Responsibilities: Analyze datasets, create dashboards, generate reports,
            identify trends and patterns, present findings to stakeholders.
            Nice to have: Machine Learning, R, Apache Spark.""",
            metadata={"source": "sample_jd_2.txt", "category": "job_descriptions", "filename": "sample_jd_2.txt"},
        ),
        Document(
            page_content="""Full Stack Developer Job Description
            Requirements: React/Angular, Node.js/Django, PostgreSQL/MongoDB, REST/GraphQL.
            Experience: 1-3 years. Strong in both frontend and backend development.
            Responsibilities: Build end-to-end features, optimize performance,
            implement responsive designs, manage databases.
            Nice to have: TypeScript, Redis, Microservices, Cloud deployment.""",
            metadata={"source": "sample_jd_3.txt", "category": "job_descriptions", "filename": "sample_jd_3.txt"},
        ),
        Document(
            page_content="""DSA Study Guide - Arrays and Strings
            Key topics: Two pointers, Sliding window, Prefix sums, Kadane's algorithm.
            Practice problems: Two Sum, Best Time to Buy/Sell Stock, Maximum Subarray,
            Longest Substring Without Repeating Characters, Container With Most Water.
            Time complexity targets: O(n) for most array problems, O(n log n) for sorting-based.
            Common patterns: Frequency counting with hashmaps, sorting + binary search.""",
            metadata={"source": "dsa_arrays.txt", "category": "dsa", "filename": "dsa_arrays.txt"},
        ),
        Document(
            page_content="""DSA Study Guide - Trees and Graphs
            Key topics: BFS, DFS, Binary Search Trees, Heaps, Topological Sort, Dijkstra.
            Practice problems: Level Order Traversal, Validate BST, Number of Islands,
            Course Schedule, Network Delay Time, Minimum Spanning Tree.
            Techniques: Recursion, iterative with stacks/queues, union-find for graphs.""",
            metadata={"source": "dsa_trees_graphs.txt", "category": "dsa", "filename": "dsa_trees_graphs.txt"},
        ),
        Document(
            page_content="""HR Interview Guide
            Common questions: Tell me about yourself, Why this company, Strengths/Weaknesses,
            Where do you see yourself in 5 years, Describe a challenging situation.
            STAR method: Situation, Task, Action, Result — use for behavioral questions.
            Tips: Research the company, prepare questions to ask, dress professionally,
            maintain eye contact, follow up with thank-you email within 24 hours.""",
            metadata={"source": "hr_guide.txt", "category": "hr_guides", "filename": "hr_guide.txt"},
        ),
        Document(
            page_content="""Learning Resources - Web Development
            Free courses: freeCodeCamp, The Odin Project, MDN Web Docs.
            Paid courses: Udemy, Coursera, Frontend Masters.
            Practice: LeetCode, HackerRank, CodePen for frontend, GitHub for portfolios.
            Projects to build: Portfolio website, Todo app, E-commerce store, Blog platform,
            Real-time chat application, Weather dashboard with API integration.""",
            metadata={"source": "web_dev_resources.txt", "category": "learning_resources", "filename": "web_dev_resources.txt"},
        ),
        Document(
            page_content="""Company Profile - TCS (Tata Consultancy Services)
            Type: IT Services & Consulting. Headquarters: Mumbai, India.
            Hiring process: Online test (aptitude + coding) → Technical interview → HR interview.
            Key skills: Java, Python, SQL, Cloud basics. Package: 3.3-7 LPA for freshers.
            Culture: Structured training program, global projects, work-life balance.
            Tips: Focus on fundamentals, TCS NQT preparation, communication skills.""",
            metadata={"source": "company_tcs.txt", "category": "company_profiles", "filename": "company_tcs.txt"},
        ),
        Document(
            page_content="""Company Profile - Infosys
            Type: IT Services & Digital Transformation. Headquarters: Bangalore, India.
            Hiring process: InfyTQ platform → Online assessment → Interview rounds.
            Key skills: Java, Python, DBMS, OOP concepts. Package: 3.6-8 LPA for freshers.
            Certifications: Infosys Springboard, InfyTQ certifications boost chances.
            Tips: Complete InfyTQ courses, practice HackerRank, strong in OOP concepts.""",
            metadata={"source": "company_infosys.txt", "category": "company_profiles", "filename": "company_infosys.txt"},
        ),
        Document(
            page_content="""Dynamic Programming Study Guide
            Key concepts: Overlapping subproblems, Optimal substructure, Memoization, Tabulation.
            Classic problems: Fibonacci, Coin Change, Longest Common Subsequence, Knapsack,
            Edit Distance, Longest Increasing Subsequence, Matrix Chain Multiplication.
            Approach: 1) Identify if DP applies 2) Define state 3) Write recurrence
            4) Choose top-down (memo) or bottom-up (table) 5) Optimize space if possible.""",
            metadata={"source": "dsa_dp.txt", "category": "dsa", "filename": "dsa_dp.txt"},
        ),
    ]
    logger.info(f"Created {len(samples)} sample documents for bootstrapping")
    return samples


# ────────────────────────────────────────────────────────────
#  CLI Entry Point
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ingestion()
