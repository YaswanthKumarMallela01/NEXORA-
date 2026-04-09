"""
NEXORA — ResumeAgent
Analyzes uploaded PDF resumes: extracts skills, matches against JDs from RAG,
and produces a structured readiness assessment.

LLM: Groq (llama-3.1-70b) with Together AI fallback
Pipeline: pdf_parser → skill_extractor → jd_matcher
Output: JSON { found_skills[], missing_skills[], match_scores{}, readiness_score, summary }
"""

import base64
import json
import logging
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
#  Step 2: Skill Extraction
# ────────────────────────────────────────────────────────────

def extract_skills(resume_text: str) -> dict:
    """Analyze resume text to extract skills, experience, and projects."""
    prompt = f"""Analyze this resume text and extract structured information.
Return ONLY valid JSON (no markdown, no ``` blocks) with this exact structure:
{{
    "technical_skills": ["skill1", "skill2"],
    "soft_skills": ["skill1", "skill2"],
    "programming_languages": ["lang1", "lang2"],
    "frameworks_tools": ["framework1", "tool1"],
    "experience_years": 0,
    "education": "degree and institution",
    "projects": [
        {{"name": "Project Name", "tech_stack": ["tech1"], "description": "brief"}}
    ],
    "certifications": ["cert1", "cert2"],
    "summary": "2-3 sentence professional summary"
}}

Resume Text:
{resume_text[:5000]}"""

    try:
        raw = _call_llm(prompt, temperature=0.1)
        cleaned = _clean_json(raw)
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Skill extraction failed: {e}")
        return {
            "technical_skills": [], "soft_skills": [],
            "programming_languages": [], "frameworks_tools": [],
            "experience_years": 0, "education": "Not specified",
            "projects": [], "certifications": [],
            "summary": "Could not parse resume details.",
        }


# ────────────────────────────────────────────────────────────
#  Step 3: JD Matching
# ────────────────────────────────────────────────────────────

def match_against_jds(skills_data: dict) -> dict:
    """Match candidate skills against job descriptions from RAG."""
    all_skills = (
        skills_data.get("technical_skills", [])
        + skills_data.get("programming_languages", [])
        + skills_data.get("frameworks_tools", [])
    )
    query = f"Job requirements matching skills: {', '.join(all_skills)}"

    jd_docs = get_job_descriptions(query, top_k=5)

    if not jd_docs:
        return {
            "match_scores": {},
            "overall_missing_skills": [],
            "message": "No job descriptions found in knowledge base. Score based on skill analysis only.",
        }

    jd_context = build_context_string(jd_docs)
    prompt = f"""Compare this candidate's skills against these job descriptions.
Return ONLY valid JSON (no markdown, no ``` blocks) with this structure:
{{
    "match_scores": {{
        "JD_1": {{"score": 72, "matched_skills": ["skill1"], "gaps": ["gap1"]}},
        "JD_2": {{"score": 65, "matched_skills": ["skill1"], "gaps": ["gap1"]}}
    }},
    "overall_missing_skills": ["skill1", "skill2"],
    "strongest_match": "JD_name",
    "weakest_areas": ["area1", "area2"]
}}

Candidate Skills:
{json.dumps(skills_data, indent=2)}

Job Descriptions:
{jd_context}"""

    try:
        raw = _call_llm(prompt, temperature=0.1)
        cleaned = _clean_json(raw)
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"JD matching failed: {e}")
        return {"match_scores": {}, "overall_missing_skills": [], "error": str(e)}


# ────────────────────────────────────────────────────────────
#  Step 4: Readiness Assessment
# ────────────────────────────────────────────────────────────

def assess_readiness(skills_data: dict, jd_match: dict) -> dict:
    """Calculate readiness score and generate final assessment."""
    prompt = f"""Based on this candidate analysis, provide a final readiness assessment.
Return ONLY valid JSON (no markdown):
{{
    "readiness_score": 0,
    "summary": "2-3 paragraph assessment",
    "strengths": ["strength1", "strength2"],
    "critical_gaps": ["gap1", "gap2"],
    "recommended_actions": ["action1", "action2"]
}}

Scoring guide:
- 0-30: Not ready — critical skill gaps
- 31-50: Needs work — significant gaps but foundation exists
- 51-70: Getting there — some gaps, focused preparation needed
- 71-85: Ready — minor gaps, strong candidate
- 86-100: Highly ready — excellent match

Skills Profile:
{json.dumps(skills_data, indent=2)[:2000]}

JD Match Results:
{json.dumps(jd_match, indent=2)[:2000]}"""

    try:
        raw = _call_llm(prompt, temperature=0.2)
        cleaned = _clean_json(raw)
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Readiness assessment failed: {e}")
        # Compute basic score from available data
        skill_count = len(skills_data.get("technical_skills", []))
        base_score = min(skill_count * 8, 60)
        return {
            "readiness_score": base_score,
            "summary": f"Found {skill_count} technical skills. Further analysis needed.",
            "strengths": skills_data.get("technical_skills", [])[:3],
            "critical_gaps": [],
            "recommended_actions": ["Upload your resume again for a complete analysis"],
        }


# ────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────

async def analyze_resume(user_id: str, pdf_base64: str) -> dict:
    """
    Full resume analysis pipeline:
    1. Parse PDF → extract text
    2. Extract skills from text
    3. Match against JDs from RAG
    4. Generate readiness assessment
    5. Save to Supabase
    """
    logger.info(f"Starting resume analysis for user {user_id}")

    try:
        # Step 1: Parse PDF
        resume_text = parse_pdf(pdf_base64)
        if resume_text.startswith("ERROR"):
            return {"success": False, "error": resume_text}

        # Step 2: Extract skills
        skills_data = extract_skills(resume_text)

        # Step 3: Match against JDs
        jd_match = match_against_jds(skills_data)

        # Step 4: Readiness assessment
        assessment = assess_readiness(skills_data, jd_match)

        # Build final result
        found_skills = (
            skills_data.get("technical_skills", [])
            + skills_data.get("programming_languages", [])
            + skills_data.get("frameworks_tools", [])
        )
        missing_skills = jd_match.get("overall_missing_skills", [])
        readiness_score = assessment.get("readiness_score", 0)

        analysis = {
            "found_skills": found_skills,
            "missing_skills": missing_skills,
            "match_scores": jd_match.get("match_scores", {}),
            "readiness_score": readiness_score,
            "summary": assessment.get("summary", ""),
            "strengths": assessment.get("strengths", []),
            "critical_gaps": assessment.get("critical_gaps", []),
            "recommended_actions": assessment.get("recommended_actions", []),
            "skills_detail": skills_data,
        }

        # Save to Supabase
        update_skill_profile(user_id, analysis)
        update_readiness_score(user_id, readiness_score)

        logger.info(f"Resume analysis complete for {user_id}: readiness={readiness_score}")
        return {"success": True, "analysis": analysis}

    except Exception as e:
        logger.error(f"Resume analysis failed for {user_id}: {e}")
        return {"success": False, "error": str(e)}


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
