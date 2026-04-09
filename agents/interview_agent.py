"""
NEXORA — InterviewAgent
Mock interview system with role-specific questions, answer evaluation,
and session scoring. Uses Gemini 1.5 Flash for large context tracking.

LLM: Google Gemini 1.5 Flash
State: Interview sessions persisted in Supabase (interview_sessions table)
Output: { question, feedback, session_score }
"""

import json
import logging
from typing import Optional, List, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from chains.orchestrator import get_gemini_llm, get_groq_llm
from db.supabase_client import (
    create_interview_session,
    get_interview_session,
    update_interview_session,
    update_user_field,
    get_user,
)

logger = logging.getLogger("nexora.agents.interview")


# ────────────────────────────────────────────────────────────
#  LLM helper — Gemini primary, Groq fallback
# ────────────────────────────────────────────────────────────

def _call_llm(prompt: str, temperature: float = 0.4) -> str:
    """Call Gemini, fall back to Groq if it fails."""
    from langchain_core.messages import HumanMessage as HM
    try:
        llm = get_gemini_llm(temperature=temperature, max_tokens=4096)
        response = llm.invoke([HM(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Gemini failed, trying Groq: {e}")
        llm = get_groq_llm(temperature=temperature, max_tokens=4096)
        response = llm.invoke([HM(content=prompt)])
        return response.content.strip()


# ────────────────────────────────────────────────────────────
#  Question Generation
# ────────────────────────────────────────────────────────────

def _generate_question(role: str, q_num: int, prev_topics: list) -> dict:
    """Generate a role-specific interview question."""
    prompt = f"""You are an expert technical interviewer for the role of {role}.
Generate interview question #{q_num}.

Previously asked topics (DO NOT repeat): {json.dumps(prev_topics)}

Rules:
- Questions 1-3: Technical (DSA, system design, or role-specific)
- Questions 4-5: Behavioral/HR (leadership, teamwork, conflict)
- Mix difficulty: 1 easy, 2 medium, 1 hard, 1 medium
- Be specific and realistic — like a real company interview

Return ONLY valid JSON (no markdown, no ```):
{{
    "question_number": {q_num},
    "question": "The interview question",
    "type": "technical or behavioral or system_design",
    "difficulty": "easy or medium or hard",
    "topic": "specific topic like arrays or react or leadership",
    "expected_duration_minutes": 5,
    "hints": ["hint1 if needed"]
}}"""

    try:
        raw = _call_llm(prompt, temperature=0.6)
        cleaned = _clean_json(raw)
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        return {
            "question_number": q_num,
            "question": f"Tell me about a challenging project you've worked on for a {role} position.",
            "type": "behavioral",
            "difficulty": "medium",
            "topic": "project experience",
            "expected_duration_minutes": 5,
            "hints": [],
        }


# ────────────────────────────────────────────────────────────
#  Answer Evaluation
# ────────────────────────────────────────────────────────────

def _evaluate_answer(question: str, answer: str, role: str, q_type: str = "technical") -> dict:
    """Evaluate a candidate's answer to an interview question."""
    prompt = f"""You are a senior technical interviewer evaluating a candidate for {role}.

Question ({q_type}): {question}
Candidate's Answer: {answer}

Evaluate the answer on these 5 dimensions (score 1-10 each):
1. Technical Accuracy — correctness of the answer
2. Communication — clarity and structure of explanation
3. Problem Solving — approach and reasoning shown
4. Relevance — how well it addresses the question
5. Depth — level of detail and understanding

Return ONLY valid JSON (no markdown, no ```):
{{
    "scores": {{
        "technical_accuracy": 7,
        "communication": 7,
        "problem_solving": 7,
        "relevance": 7,
        "depth": 7,
        "average": 7.0
    }},
    "feedback": {{
        "strengths": ["strength1", "strength2"],
        "improvements": ["area1", "area2"],
        "ideal_answer_summary": "brief description of what a great answer looks like",
        "overall_comment": "1-2 sentences of encouragement and guidance"
    }},
    "pass": true
}}

A pass requires average >= 6.0. Be fair but rigorous."""

    try:
        raw = _call_llm(prompt, temperature=0.2)
        cleaned = _clean_json(raw)
        result = json.loads(cleaned)

        # Calculate average if not provided
        scores = result.get("scores", {})
        dims = ["technical_accuracy", "communication", "problem_solving", "relevance", "depth"]
        values = [scores.get(d, 5) for d in dims]
        scores["average"] = round(sum(values) / len(values), 2)
        result["scores"] = scores
        result["pass"] = scores["average"] >= 6.0
        return result

    except Exception as e:
        logger.error(f"Answer evaluation failed: {e}")
        return {
            "scores": {"technical_accuracy": 5, "communication": 5, "problem_solving": 5,
                       "relevance": 5, "depth": 5, "average": 5.0},
            "feedback": {"strengths": [], "improvements": ["Could not fully evaluate"],
                         "overall_comment": "Please try again with a more detailed answer."},
            "pass": False,
        }


# ────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────

async def start_interview(user_id: str, role: str) -> dict:
    """
    Start a new mock interview session.
    Creates session in DB and generates the first question.
    """
    logger.info(f"Starting interview for user {user_id}, role: {role}")

    # Create session in DB
    session = create_interview_session({
        "user_id": user_id,
        "role": role,
        "questions": [],
        "answers": [],
        "scores": {"per_question": [], "running_average": 0},
    })

    if not session:
        return {"success": False, "error": "Failed to create interview session"}

    # Generate first question
    question_data = _generate_question(role, 1, [])

    # Update session
    update_interview_session(session["id"], {
        "questions": [question_data],
    })

    return {
        "success": True,
        "session_id": session["id"],
        "role": role,
        "response": f"Welcome to your mock interview for **{role}**! I'll ask you 5 questions — 3 technical and 2 behavioral. Take your time with each answer.\n\nHere's your first question:",
        "question": question_data,
        "question_number": 1,
        "total_questions": 5,
    }


async def submit_answer(user_id: str, session_id: str, answer: str) -> dict:
    """
    Submit an answer to the current interview question.
    Evaluates the answer and generates the next question (or summary).
    """
    logger.info(f"Answer submitted for session {session_id}")

    session = get_interview_session(session_id)
    if not session:
        return {"success": False, "error": "Session not found"}

    questions = session.get("questions", [])
    answers = session.get("answers", [])
    scores = session.get("scores", {"per_question": [], "running_average": 0})

    current_q_num = len(questions)
    if current_q_num == 0:
        return {"success": False, "error": "No question to answer"}

    is_last = current_q_num >= 5
    current_question = questions[-1]

    # Add answer
    answers.append({"question_number": current_q_num, "answer": answer})

    # Evaluate the answer
    q_text = current_question.get("question", "") if isinstance(current_question, dict) else str(current_question)
    q_type = current_question.get("type", "technical") if isinstance(current_question, dict) else "technical"
    role = session.get("role", "Software Engineer")

    eval_data = _evaluate_answer(q_text, answer, role, q_type)

    # Update scores
    scores["per_question"].append(eval_data.get("scores", {}))
    all_avgs = [s.get("average", 5) for s in scores["per_question"]]
    scores["running_average"] = round(sum(all_avgs) / len(all_avgs), 2)

    # Generate next question or summary
    next_question = None
    response_text = ""

    if not is_last:
        prev_topics = [q.get("topic", "") for q in questions if isinstance(q, dict)]
        next_question = _generate_question(role, current_q_num + 1, prev_topics)
        questions.append(next_question)

        avg_score = eval_data.get("scores", {}).get("average", 5)
        feedback = eval_data.get("feedback", {})
        strengths = ", ".join(feedback.get("strengths", [])[:2])
        improvements = ", ".join(feedback.get("improvements", [])[:2])

        response_text = f"""**Score: {avg_score}/10**

{'✅ **Strengths:** ' + strengths if strengths else ''}
{'⚠️ **To improve:** ' + improvements if improvements else ''}

{feedback.get('overall_comment', '')}

---

**Question {current_q_num + 1} of 5:**"""
    else:
        # Last question — generate summary
        feedback = eval_data.get("feedback", {})
        final_score = scores["running_average"]
        readiness = "Ready for Interviews! 🎉" if final_score >= 7 else "Needs More Practice 📚" if final_score >= 5 else "Keep Practicing 💪"

        response_text = f"""**Final Answer Score: {eval_data.get('scores', {}).get('average', 5)}/10**

{feedback.get('overall_comment', '')}

---

## 📊 Interview Complete!

**Overall Score: {final_score}/10**
**Verdict: {readiness}**

**Per-question scores:**
"""
        for i, qs in enumerate(scores["per_question"]):
            response_text += f"  Q{i+1}: {qs.get('average', 'N/A')}/10\n"

        # Save to user profile
        _save_final_score(user_id, session_id, scores)

    # Update session in DB
    update_interview_session(session_id, {
        "questions": questions,
        "answers": answers,
        "scores": scores,
    })

    return {
        "success": True,
        "session_id": session_id,
        "response": response_text,
        "feedback": eval_data,
        "next_question": next_question,
        "question_number": current_q_num + (0 if is_last else 1),
        "session_score": scores.get("running_average", 0),
        "is_complete": is_last,
    }


async def get_session_summary(session_id: str) -> dict:
    """Get the full summary of a completed interview session."""
    session = get_interview_session(session_id)
    if not session:
        return {"success": False, "error": "Session not found"}

    scores = session.get("scores", {})
    questions = session.get("questions", [])
    answers = session.get("answers", [])

    qa_pairs = []
    per_q_scores = scores.get("per_question", [])
    for i, q in enumerate(questions):
        pair = {
            "question": q,
            "answer": answers[i] if i < len(answers) else None,
            "score": per_q_scores[i] if i < len(per_q_scores) else None,
        }
        qa_pairs.append(pair)

    return {
        "success": True,
        "session_id": session_id,
        "role": session.get("role"),
        "total_questions": len(questions),
        "total_answered": len(answers),
        "session_score": scores.get("running_average", 0),
        "qa_pairs": qa_pairs,
        "created_at": session.get("created_at"),
    }


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────

def _save_final_score(user_id: str, session_id: str, scores: dict):
    """Save final interview score to user's interview_scores array."""
    user = get_user(user_id) or {}
    existing_scores = user.get("interview_scores", [])
    if not isinstance(existing_scores, list):
        existing_scores = []

    existing_scores.append({
        "session_id": session_id,
        "score": scores.get("running_average", 0),
        "per_question": scores.get("per_question", []),
    })

    update_user_field(user_id, "interview_scores", existing_scores)
    logger.info(f"Final score saved for {user_id}: {scores.get('running_average', 0)}")


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
