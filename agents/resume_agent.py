"""
NEXORA — ResumeAgent
Analyzes uploaded PDF resumes: extracts skills, matches against JDs from RAG,
and produces a structured readiness assessment.

Optimized for Render (Single LLM Call).
"""

import base64
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict

import fitz  # PyMuPDF

from langchain_core.messages import HumanMessage

from chains.orchestrator import get_groq_llm, get_together_llm
from rag.retriever import get_job_descriptions, build_context_string
from db.supabase_client import update_skill_profile, update_readiness_score

logger = logging.getLogger("nexora.agents.resume")


# ────────────────────────────────────────────────────────────
#  LLM helper
# ────────────────────────────────────────────────────────────

def _call_llm(prompt: str, temperature: float = 0.1) -> str:
    """Call Groq, fall back to Together AI."""
    try:
        llm = get_groq_llm(temperature=temperature, max_tokens=4096)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Groq failed, trying Together AI: {e}")
        llm = get_together_llm(temperature=temperature, max_tokens=4096)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()


# ────────────────────────────────────────────────────────────
#  Step 1: PDF Parser
# ────────────────────────────────────────────────────────────

def parse_pdf(pdf_base64: str) -> str:
    """Extract text content from a base64-encoded PDF file."""
    doc = None
    try:
        raw = (pdf_base64 or "").strip()
        if raw.startswith("data:"):
            idx = raw.find("base64,")
            if idx != -1:
                raw = raw[idx + 7 :]
        # Whitespace/newlines in pasted base64 breaks decode on some clients
        raw = "".join(raw.split())
        pdf_bytes = base64.b64decode(raw, validate=False)
        if not pdf_bytes:
            return "ERROR: Empty PDF data after decoding."

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n_pages = doc.page_count
        text_parts: List[str] = []
        for page in doc:
            text_parts.append(page.get_text() or "")
        text = "".join(text_parts)
    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        return f"ERROR: Failed to parse PDF — {str(e)}"
    finally:
        if doc is not None:
            doc.close()

    if not text.strip():
        return "ERROR: Could not extract text from PDF. The file may be image-based or scanned."

    logger.info(f"Extracted {len(text)} characters from PDF ({n_pages} pages)")
    return text.strip()


# ────────────────────────────────────────────────────────────
#  Unified Analysis Pipeline (Render Optimized)
# ────────────────────────────────────────────────────────────

async def analyze_resume(user_id: str, pdf_base64: str) -> dict:
    """
    Unified resume analysis pipeline (Optimized for Render's 10s timeout).
    1. Parse PDF → extract text
    2. Query RAG for JDs
    3. ONE LLM call for extraction + matching + scoring + proficiency
    4. Save to Supabase
    """
    logger.info(f"Starting unified resume analysis for user {user_id}")

    try:
        # Step 1: Parse PDF
        resume_text = parse_pdf(pdf_base64)
        if resume_text.startswith("ERROR"):
            return {"success": False, "error": resume_text}

        # Step 2: Query RAG for context (Fast local operation)
        jd_docs = get_job_descriptions("software engineer developer data scientist", top_k=3)
        jd_context = build_context_string(jd_docs)

        # Step 3: Single Unified LLM Call
        prompt = f"""You are a senior technical recruiter and talent analyst.
Analyze this resume text and provide a comprehensive intelligence report.

KNOWLEDGE BASE (Top Matching Job Descriptions):
{jd_context}

RESUME TEXT:
{resume_text[:6000]}

RULES:
1. Provide a detailed readiness assessment (0-100).
2. Extract technical skills with PROFICIENCY LEVELS (0-100). This is critical for radar charts.
3. Compare against the provided JDs from the Knowledge Base.
4. BE REALISTIC. Do not provide generic scores. A score of 80 must be earned.
5. Identify categorical groups for skills (e.g. Languages, Frameworks, Tools).

RETURN ONLY VALID JSON:
{{
    "readiness_score": 75,
    "summary": "assessment summary",
    "skills_matrix": [
        {{"name": "Python", "category": "Languages", "proficiency": 90, "matched_in_jd": true}},
        {{"name": "React", "category": "Frameworks", "proficiency": 65, "matched_in_jd": true}}
    ],
    "strengths": ["string1"],
    "critical_gaps": ["gap1"],
    "recommended_actions": ["action1"],
    "jd_match_details": {{
        "overall_fit": "High/Medium/Low",
        "missing_critical_skills": ["skill1"]
    }}
}}
"""

        raw = _call_llm(prompt, temperature=0.1)
        cleaned = _clean_json(raw)
        analysis_data = json.loads(cleaned)

        # Build final backward-compatible result for dashboard
        readiness_score = analysis_data.get("readiness_score", 0)
        
        # Format skills for current dashboard + future Matrix
        skills_matrix = analysis_data.get("skills_matrix", [])
        found_skills = [s["name"] for s in skills_matrix]
        
        # New: Skill profile with proficiencies
        skill_profile = {
            "skills": skills_matrix,
            "readiness": readiness_score,
            "summary": analysis_data.get("summary", ""),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        analysis_result = {
            "found_skills": found_skills,
            "missing_skills": analysis_data.get("jd_match_details", {}).get("missing_critical_skills", []),
            "readiness_score": readiness_score,
            "summary": analysis_data.get("summary", ""),
            "strengths": analysis_data.get("strengths", []),
            "critical_gaps": analysis_data.get("critical_gaps", []),
            "recommended_actions": analysis_data.get("recommended_actions", []),
            "skills_matrix": skills_matrix,
            "jd_analysis": analysis_data.get("jd_match_details", {})
        }

        # Save to Supabase
        update_skill_profile(user_id, skill_profile)
        update_readiness_score(user_id, readiness_score)

        logger.info(f"Unified analysis complete for {user_id}: readiness={readiness_score}")
        return {"success": True, "analysis": analysis_result}

    except Exception as e:
        logger.error(f"Unified analysis failed for {user_id}: {e}")
        return {"success": False, "error": f"Intelligence Agency Timeout or Error — {str(e)}"}


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────

def _clean_json(text: str) -> str:
    """Extract clean JSON from LLM output."""
    if "```" in text:
        blocks = text.split("```")
        for block in blocks:
            clean = block.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()
            try:
                json.loads(clean)
                return clean
            except json.JSONDecodeError:
                continue

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start: end + 1]
    return text
