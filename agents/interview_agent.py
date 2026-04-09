"""
NEXORA — InterviewAgent
Mock interview system with role-specific questions, answer evaluation,
and session scoring. Uses Gemini 1.5 Flash for large context tracking.

LLM: Google Gemini 1.5 Flash
State: Interview sessions persisted in Supabase (interview_sessions table)
Tools: question_generator, answer_evaluator
Output: { question, feedback, session_score }
"""

import json
import logging
from typing import Optional, List, Dict

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

from chains.orchestrator import get_agent_llm
from db.supabase_client import (
    create_interview_session,
    get_interview_session,
    update_interview_session,
    update_user_field,
    get_user,
)

logger = logging.getLogger("nexora.agents.interview")


# ────────────────────────────────────────────────────────────
#  Tools
# ────────────────────────────────────────────────────────────

@tool
def question_generator(context: str) -> str:
    """
    Generate a role-specific interview question based on the interview context.
    Input: JSON string with keys: role, question_number, previous_topics (list of already asked topics).
    Output: JSON with the generated question and its metadata.
    """
    try:
        ctx = json.loads(context)
    except json.JSONDecodeError:
        ctx = {"role": context, "question_number": 1, "previous_topics": []}

    role = ctx.get("role", "Software Engineer")
    q_num = ctx.get("question_number", 1)
    prev_topics = ctx.get("previous_topics", [])

    llm = get_agent_llm("interview", temperature=0.6)

    prompt = f"""You are an expert technical interviewer for the role of {role}.
Generate interview question #{q_num}.

Previously asked topics (DO NOT repeat): {json.dumps(prev_topics)}

Rules:
- Questions 1-3: Technical (DSA, system design, or role-specific)
- Questions 4-5: Behavioral/HR (leadership, teamwork, conflict)
- Mix difficulty: 1 easy, 2 medium, 1 hard, 1 medium
- Be specific and realistic — like a real company interview

Return ONLY valid JSON:
{{
    "question_number": {q_num},
    "question": "The interview question",
    "type": "technical|behavioral|system_design",
    "difficulty": "easy|medium|hard",
    "topic": "specific topic (e.g., arrays, react, leadership)",
    "expected_duration_minutes": 5,
    "hints": ["hint1 if needed"]
}}"""

    try:
        response = llm.invoke(prompt)
        content = _clean_json(response.content.strip())
        json.loads(content)
        return content
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        return json.dumps({
            "question_number": q_num,
            "question": f"Tell me about a challenging project you've worked on for a {role} position.",
            "type": "behavioral",
            "difficulty": "medium",
            "topic": "project experience",
            "expected_duration_minutes": 5,
            "hints": [],
        })


@tool
def answer_evaluator(evaluation_input: str) -> str:
    """
    Evaluate a candidate's answer to an interview question.
    Scores on 5 dimensions and provides structured feedback.
    Input: JSON string with keys: question, answer, role, question_type.
    Output: JSON with scores and detailed feedback.
    """
    try:
        data = json.loads(evaluation_input)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid input JSON"})

    question = data.get("question", "")
    answer = data.get("answer", "")
    role = data.get("role", "Software Engineer")
    q_type = data.get("question_type", "technical")

    llm = get_agent_llm("interview", temperature=0.2)

    prompt = f"""You are a senior technical interviewer evaluating a candidate for {role}.

Question ({q_type}): {question}
Candidate's Answer: {answer}

Evaluate the answer on these 5 dimensions (score 1-10 each):
1. Technical Accuracy — correctness of the answer
2. Communication — clarity and structure of explanation
3. Problem Solving — approach and reasoning shown
4. Relevance — how well it addresses the question
5. Depth — level of detail and understanding

Return ONLY valid JSON:
{{
    "scores": {{
        "technical_accuracy": 0,
        "communication": 0,
        "problem_solving": 0,
        "relevance": 0,
        "depth": 0,
        "average": 0.0
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
        response = llm.invoke(prompt)
        content = _clean_json(response.content.strip())

        result = json.loads(content)

        # Calculate average if not provided
        scores = result.get("scores", {})
        dims = ["technical_accuracy", "communication", "problem_solving", "relevance", "depth"]
        values = [scores.get(d, 5) for d in dims]
        scores["average"] = round(sum(values) / len(values), 2)
        result["scores"] = scores
        result["pass"] = scores["average"] >= 6.0

        return json.dumps(result)
    except Exception as e:
        logger.error(f"Answer evaluation failed: {e}")
        return json.dumps({
            "scores": {"average": 5.0},
            "feedback": {"overall_comment": "Evaluation encountered an error. Please try again."},
            "error": str(e),
        })


# ────────────────────────────────────────────────────────────
#  Agent Setup
# ────────────────────────────────────────────────────────────

INTERVIEW_SYSTEM_PROMPT = """You are NEXORA's Interview Agent — a professional mock interviewer.

You are conducting a structured mock interview for the role of: {role}

YOUR BEHAVIOR:
- You ask ONE question at a time, then wait for the answer
- After each answer, you evaluate it using answer_evaluator
- You maintain a running score throughout the session
- You adapt difficulty based on performance
- Standard session: 5 questions (3 technical + 2 behavioral)

STUDENT PROFILE:
{student_profile}

SESSION STATE:
Questions asked so far: {questions_asked}
Current question number: {current_question}
Running score: {running_score}

After evaluation, present the feedback in a clear, encouraging format.
Always end with the next question OR the session summary if all questions are done."""


def create_interview_agent(role: str, student_profile: dict, session_state: dict) -> AgentExecutor:
    """Create an InterviewAgent for a specific role and session."""
    llm = get_agent_llm("interview", temperature=0.4, max_tokens=8192)
    tools = [question_generator, answer_evaluator]

    questions_asked = session_state.get("questions_asked", [])
    current_q = session_state.get("current_question", 1)
    running_score = session_state.get("running_score", "N/A")

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTERVIEW_SYSTEM_PROMPT.format(
            role=role,
            student_profile=json.dumps(student_profile),
            questions_asked=json.dumps(questions_asked),
            current_question=str(current_q),
            running_score=str(running_score),
        )),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


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

    # Generate first question using agent
    user = get_user(user_id) or {}
    session_state = {"questions_asked": [], "current_question": 1, "running_score": "N/A"}
    agent = create_interview_agent(role, user.get("skill_profile", {}), session_state)

    try:
        result = agent.invoke({
            "input": f"Start the interview. Generate the first question for a {role} position."
        })

        output = result.get("output", "")

        # Try to extract question JSON from tool results
        question_data = _extract_tool_result(result, "question_generator")

        # Update session with first question
        if question_data:
            update_interview_session(session["id"], {
                "questions": [question_data],
            })

        return {
            "success": True,
            "session_id": session["id"],
            "role": role,
            "response": output,
            "question": question_data,
            "question_number": 1,
            "total_questions": 5,
        }

    except Exception as e:
        logger.error(f"Failed to start interview: {e}")
        return {"success": False, "error": str(e), "session_id": session.get("id")}


async def submit_answer(
    user_id: str,
    session_id: str,
    answer: str,
) -> dict:
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
    is_last = current_q_num >= 5

    # Add answer to session
    answers.append({"question_number": current_q_num, "answer": answer})

    user = get_user(user_id) or {}
    role = session.get("role", "Software Engineer")

    # Build context for the agent
    prev_topics = [q.get("topic", "") for q in questions if isinstance(q, dict)]
    session_state = {
        "questions_asked": prev_topics,
        "current_question": current_q_num + 1,
        "running_score": scores.get("running_average", "N/A"),
    }

    agent = create_interview_agent(role, user.get("skill_profile", {}), session_state)

    if is_last:
        input_text = f"""Evaluate this answer and then provide a FINAL SESSION SUMMARY.

Question: {json.dumps(questions[-1]) if questions else 'N/A'}
Answer: {answer}

This was the last question. After evaluation, summarize the entire session:
- Overall score
- Key strengths
- Areas to improve
- Final recommendation (Ready / Needs Practice / Not Ready)"""
    else:
        input_text = f"""Evaluate this answer, then generate the NEXT question.

Question: {json.dumps(questions[-1]) if questions else 'N/A'}
Answer: {answer}
Previous topics covered: {json.dumps(prev_topics)}
Next question number: {current_q_num + 1}"""

    try:
        result = agent.invoke({"input": input_text})

        output = result.get("output", "")

        # Extract evaluation scores from tool results
        eval_data = _extract_tool_result(result, "answer_evaluator")
        if eval_data:
            scores["per_question"].append(eval_data.get("scores", {}))
            all_avgs = [s.get("average", 5) for s in scores["per_question"]]
            scores["running_average"] = round(sum(all_avgs) / len(all_avgs), 2)

        # Extract next question if not last
        next_question = None
        if not is_last:
            next_question = _extract_tool_result(result, "question_generator")
            if next_question:
                questions.append(next_question)

        # Update session in DB
        update_interview_session(session_id, {
            "questions": questions,
            "answers": answers,
            "scores": scores,
        })

        # If last question, save final score to user profile
        if is_last:
            _save_final_score(user_id, session_id, scores)

        return {
            "success": True,
            "session_id": session_id,
            "response": output,
            "feedback": eval_data,
            "next_question": next_question,
            "question_number": current_q_num + (0 if is_last else 1),
            "session_score": scores.get("running_average", 0),
            "is_complete": is_last,
        }

    except Exception as e:
        logger.error(f"Answer evaluation failed: {e}")
        return {"success": False, "error": str(e)}


async def get_session_summary(session_id: str) -> dict:
    """Get the full summary of a completed interview session."""
    session = get_interview_session(session_id)
    if not session:
        return {"success": False, "error": "Session not found"}

    scores = session.get("scores", {})
    questions = session.get("questions", [])
    answers = session.get("answers", [])

    # Build Q&A pairs
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

def _extract_tool_result(result: dict, tool_name: str) -> Optional[dict]:
    """Extract parsed tool output from agent intermediate steps."""
    steps = result.get("intermediate_steps", [])
    for step in steps:
        action, observation = step
        if action.tool == tool_name:
            try:
                return json.loads(observation)
            except (json.JSONDecodeError, TypeError):
                pass
    return None


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
