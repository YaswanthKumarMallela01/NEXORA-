"""
NEXORA — CoachAgent (Stateful)
Persistent AI placement mentor with conversation memory.
Assigns tasks, generates roadmaps, recommends resources via RAG.

LLM: Groq (llama-3.1-70b) with Together AI fallback
Memory: ConversationBufferWindowMemory (k=20), persisted in Supabase
Tools: task_assigner, roadmap_generator, resource_recommender
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from chains.orchestrator import get_groq_llm, get_together_llm
from db.supabase_client import (
    get_user,
    get_coach_memory,
    update_coach_memory,
    create_task,
    get_tasks,
    get_overdue_tasks,
    save_roadmap,
)

logger = logging.getLogger("nexora.agents.coach")


# ────────────────────────────────────────────────────────────
#  Memory Management
# ────────────────────────────────────────────────────────────

def _load_memory(user_id: str) -> list:
    """Load conversation memory from Supabase as message list."""
    stored = get_coach_memory(user_id)
    messages = []
    for entry in stored:
        if entry.get("role") == "human":
            messages.append(HumanMessage(content=entry["content"]))
        elif entry.get("role") == "ai":
            messages.append(AIMessage(content=entry["content"]))
    # Keep last 20 pairs (40 messages)
    return messages[-40:]


def _save_memory(user_id: str, messages: list):
    """Persist conversation memory to Supabase."""
    serialized = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            serialized.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serialized.append({"role": "ai", "content": msg.content})
    # Keep only last 40 messages
    serialized = serialized[-40:]
    update_coach_memory(user_id, serialized)


# ────────────────────────────────────────────────────────────
#  Task Helper
# ────────────────────────────────────────────────────────────

def _auto_assign_tasks(user_id: str, response_text: str):
    """Parse agent response for actionable tasks and auto-assign them."""
    # Simple heuristic: if the response contains numbered items that look like tasks
    lines = response_text.split("\n")
    tasks_created = 0
    for line in lines:
        line = line.strip()
        # Match lines like "1. Do X", "- Complete Y", "* Practice Z"
        if tasks_created >= 3:
            break  # Max 3 auto-assigned tasks per response
        for prefix in ["1.", "2.", "3.", "- ", "* ", "• "]:
            if line.startswith(prefix):
                task_text = line[len(prefix):].strip()
                # Only create if it looks like an actionable task (>10 chars, <200 chars)
                if 10 < len(task_text) < 200 and any(verb in task_text.lower() for verb in
                    ["complete", "practice", "solve", "study", "learn", "build", "review", "prepare", "read", "watch"]):
                    try:
                        create_task({
                            "user_id": user_id,
                            "title": task_text[:100],
                            "due_date": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                            "assigned_by": "coach",
                        })
                        tasks_created += 1
                    except Exception as e:
                        logger.warning(f"Auto-task creation failed: {e}")
                break


# ────────────────────────────────────────────────────────────
#  Core Chat Function
# ────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are NEXORA — a strict but supportive AI placement mentor.

YOUR PERSONALITY:
- You are NOT a generic chatbot. You are a decisive, experienced placement officer.
- You KNOW the student's complete history (skills, tasks, interview scores).
- You MAKE DECISIONS — you don't just suggest. You assign tasks, set deadlines, and escalate.
- You are encouraging but firm. If a student is slacking, you call it out directly.
- You speak with authority but genuine care.

CRITICAL BEHAVIORS:
- If student mentions they missed tasks → Create a recovery plan with specific tasks
- If student asks "what should I do?" → Give a concrete weekly plan with numbered tasks
- If student asks about a skill → Provide specific resources and practice problems
- Always reference their skill profile when giving advice
- Never say "you could" or "you might want to" — say "Here's what you're doing this week:"
- When assigning tasks, format them as numbered items so they can be auto-tracked

STUDENT CONTEXT:
Name: {name}
Readiness Score: {readiness_score}/100
Skills: {skills}
Missing Skills: {missing_skills}
At Risk: {at_risk}

OVERDUE TASKS:
{overdue_tasks}"""


async def chat_with_coach(user_id: str, message: str) -> dict:
    """
    Send a message to the CoachAgent and get a response.
    Memory is automatically loaded and saved.
    Uses direct LLM invocation (more reliable than AgentExecutor).
    """
    logger.info(f"Coach chat for user {user_id}: {message[:100]}...")

    try:
        # Load user context
        user = get_user(user_id) or {}
        skill_profile = user.get("skill_profile", {})
        overdue = get_overdue_tasks(user_id)

        overdue_text = "None"
        if overdue:
            overdue_text = "\n".join([f"- {t['title']} (due: {t.get('due_date', 'N/A')})" for t in overdue[:5]])

        system_prompt = SYSTEM_TEMPLATE.format(
            name=user.get("name", "Student"),
            readiness_score=user.get("readiness_score", 0),
            skills=json.dumps(skill_profile.get("found_skills", [])),
            missing_skills=json.dumps(skill_profile.get("missing_skills", [])),
            at_risk=user.get("at_risk", False),
            overdue_tasks=overdue_text,
        )

        # Load conversation memory
        memory_messages = _load_memory(user_id)

        # Build full message list
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(memory_messages)
        messages.append(HumanMessage(content=message))

        # Call LLM (Groq primary, Together fallback)
        try:
            llm = get_groq_llm(temperature=0.5, max_tokens=4096)
            response = llm.invoke(messages)
        except Exception as groq_err:
            logger.warning(f"Groq failed, trying Together AI: {groq_err}")
            llm = get_together_llm(temperature=0.5, max_tokens=4096)
            response = llm.invoke(messages)

        reply = response.content

        # Update memory with new exchange
        memory_messages.append(HumanMessage(content=message))
        memory_messages.append(AIMessage(content=reply))
        _save_memory(user_id, memory_messages)

        # Auto-assign tasks from response
        _auto_assign_tasks(user_id, reply)

        return {
            "success": True,
            "response": reply,
            "tool_calls": [],
        }

    except Exception as e:
        logger.error(f"Coach chat failed for {user_id}: {e}")
        return {
            "success": False,
            "response": "I encountered an issue processing your request. Please try again.",
            "error": str(e),
        }


async def get_conversation_history(user_id: str) -> list:
    """Fetch the full conversation history for a user."""
    return get_coach_memory(user_id)
