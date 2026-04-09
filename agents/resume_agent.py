"""
NEXORA — ResumeAgent
Analyzes uploaded PDF resumes: extracts skills, matches against JDs from RAG,
and produces a structured readiness assessment.

LLM: Groq (llama-3.1-70b) with Together AI fallback
Tools: pdf_parser, skill_extractor, jd_matcher
Output: JSON { found_skills[], missing_skills[], match_scores{}, readiness_score, summary }
"""

import base64
import json
import logging
from typing import Optional, List, Dict

import fitz  # PyMuPDF
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

from chains.orchestrator import get_agent_llm
from rag.retriever import get_job_descriptions, build_context_string
from db.supabase_client import update_skill_profile, update_readiness_score

logger = logging.getLogger("nexora.agents.resume")


# ────────────────────────────────────────────────────────────
#  Tools
# ────────────────────────────────────────────────────────────

@tool
def pdf_parser(pdf_base64: str) -> str:
    """
    Extract text content from a base64-encoded PDF file.
    Input: base64 string of the PDF file.
    Output: extracted text content from all pages.
    """
    try:
        pdf_bytes = base64.b64decode(pdf_base64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        if not text.strip():
            return "ERROR: Could not extract text from PDF. The file may be image-based."

        logger.info(f"Extracted {len(text)} characters from PDF ({doc.page_count} pages)")
        return text.strip()
    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        return f"ERROR: Failed to parse PDF — {str(e)}"


@tool
def skill_extractor(resume_text: str) -> str:
    """
    Analyze resume text to extract skills, experience, and projects.
    Input: raw text extracted from a resume.
    Output: JSON with categorized skills, experience summary, and projects.
    """
    llm = get_agent_llm("resume", temperature=0.1)

    prompt = f"""Analyze this resume text and extract structured information.
Return ONLY valid JSON (no markdown, no code blocks) with this exact structure:
{{
    "technical_skills": ["skill1", "skill2", ...],
    "soft_skills": ["skill1", "skill2", ...],
    "programming_languages": ["lang1", "lang2", ...],
    "frameworks_tools": ["framework1", "tool1", ...],
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
        response = llm.invoke(prompt)
        content = response.content.strip()
        content = _clean_json(content)
        # Validate JSON
        json.loads(content)
        return content
    except json.JSONDecodeError:
        return json.dumps({
            "technical_skills": [],
            "soft_skills": [],
            "programming_languages": [],
            "frameworks_tools": [],
            "experience_years": 0,
            "education": "Not specified",
            "projects": [],
            "certifications": [],
            "summary": "Could not parse resume details.",
        })
    except Exception as e:
        logger.error(f"Skill extraction failed: {e}")
        return json.dumps({"error": str(e)})


@tool
def jd_matcher(skills_json: str) -> str:
    """
    Fetch top 5 job descriptions from RAG and compare candidate skills against them.
    Input: JSON string of extracted skills from skill_extractor.
    Output: JSON with match scores per JD and missing skills.
    """
    try:
        skills_data = json.loads(skills_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid skills JSON input"})

    # Build a query from the candidate's skills
    all_skills = (
        skills_data.get("technical_skills", [])
        + skills_data.get("programming_languages", [])
        + skills_data.get("frameworks_tools", [])
    )
    query = f"Job requirements matching skills: {', '.join(all_skills)}"

    # Fetch JDs from RAG
    jd_docs = get_job_descriptions(query, top_k=5)

    if not jd_docs:
        return json.dumps({
            "match_scores": {},
            "missing_skills": [],
            "message": "No job descriptions found in knowledge base.",
        })

    # Use LLM to compare skills against JDs
    llm = get_agent_llm("resume", temperature=0.1)

    jd_context = build_context_string(jd_docs)
    prompt = f"""Compare this candidate's skills against these job descriptions.
Return ONLY valid JSON (no markdown) with this structure:
{{
    "match_scores": {{
        "JD_source_1": {{"score": 0, "matched_skills": [...], "gaps": [...]}},
        "JD_source_2": {{"score": 0, "matched_skills": [...], "gaps": [...]}}
    }},
    "overall_missing_skills": ["skill1", "skill2"],
    "strongest_match": "JD_source_name",
    "weakest_areas": ["area1", "area2"]
}}

Candidate Skills:
{json.dumps(skills_data, indent=2)}

Job Descriptions:
{jd_context}"""

    try:
        response = llm.invoke(prompt)
        content = _clean_json(response.content.strip())
        json.loads(content)
        return content
    except Exception as e:
        logger.error(f"JD matching failed: {e}")
        return json.dumps({"error": str(e), "match_scores": {}})


# ────────────────────────────────────────────────────────────
#  Agent Setup
# ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are NEXORA's Resume Analysis Agent — a precision placement readiness assessor.

Your job:
1. Parse the uploaded PDF resume using the pdf_parser tool
2. Extract all skills, experience, and projects using skill_extractor
3. Match the candidate's profile against real job descriptions using jd_matcher
4. Synthesize everything into a final readiness assessment

You MUST call all three tools in sequence: pdf_parser → skill_extractor → jd_matcher.

After getting all tool results, provide a final JSON response with this EXACT structure:
{{
    "found_skills": ["skill1", "skill2", ...],
    "missing_skills": ["skill1", "skill2", ...],
    "match_scores": {{"jd_name": score, ...}},
    "readiness_score": 0-100,
    "summary": "A comprehensive 2-3 paragraph readiness assessment"
}}

Be thorough, precise, and constructive. The readiness_score should reflect:
- 0-30: Not ready — critical skill gaps
- 31-50: Needs work — significant gaps but foundation exists
- 51-70: Getting there — some gaps, focused preparation needed
- 71-85: Ready — minor gaps, strong candidate
- 86-100: Highly ready — excellent match across JDs"""


def create_resume_agent() -> AgentExecutor:
    """Create and return the ResumeAgent with all tools."""
    llm = get_agent_llm("resume", temperature=0.2)
    tools = [pdf_parser, skill_extractor, jd_matcher]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────

async def analyze_resume(user_id: str, pdf_base64: str) -> dict:
    """
    Full resume analysis pipeline.

    1. Runs ResumeAgent (pdf_parser → skill_extractor → jd_matcher)
    2. Parses structured output
    3. Saves results to Supabase
    4. Returns analysis results
    """
    logger.info(f"Starting resume analysis for user {user_id}")

    agent = create_resume_agent()

    try:
        result = agent.invoke({
            "input": f"Analyze this resume (base64-encoded PDF): {pdf_base64}"
        })

        output = result.get("output", "")

        # Try to parse JSON from agent output
        analysis = _extract_json(output)

        if analysis:
            # Save to Supabase
            update_skill_profile(user_id, analysis)

            readiness = analysis.get("readiness_score", 0)
            update_readiness_score(user_id, readiness)

            logger.info(f"Resume analysis complete for {user_id}: readiness={readiness}")
            return {
                "success": True,
                "analysis": analysis,
                "intermediate_steps": len(result.get("intermediate_steps", [])),
            }
        else:
            # Fallback: return raw output
            return {
                "success": True,
                "analysis": {"summary": output, "readiness_score": 0},
                "raw_output": output,
            }

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


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from agent output, handling various formats."""
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try cleaning
    try:
        cleaned = _clean_json(text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None
