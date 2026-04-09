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

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory

from chains.orchestrator import get_agent_llm
from rag.retriever import get_learning_resources, get_dsa_resources, build_context_string
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
#  Tool Context (set per-request via module-level variable)
# ────────────────────────────────────────────────────────────

_current_user_id: Optional[str] = None


def _set_context(user_id: str):
    """Set the current user context for tools."""
    global _current_user_id
    _current_user_id = user_id


# ────────────────────────────────────────────────────────────
#  Tools
# ────────────────────────────────────────────────────────────

@tool
def task_assigner(task_json: str) -> str:
    """
    Assign a task to the current student. Creates the task in the database.
    Input: JSON string with keys: title (required), due_date (ISO format, optional),
           status (default: 'pending').
    Output: Confirmation with task details.
    Example input: '{"title": "Complete array problems on LeetCode", "due_date": "2026-04-15T23:59:00Z"}'
    """
    try:
        task_data = json.loads(task_json)
    except json.JSONDecodeError:
        # Handle plain text input
        task_data = {"title": task_json}

    if not _current_user_id:
        return "ERROR: No user context set."

    # Set default due date if not provided (7 days from now)
    if "due_date" not in task_data or not task_data["due_date"]:
        default_due = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        task_data["due_date"] = default_due

    task_data["user_id"] = _current_user_id
    task_data["assigned_by"] = "coach"

    result = create_task(task_data)
    logger.info(f"Task assigned to {_current_user_id}: {task_data['title']}")

    return json.dumps({
        "status": "assigned",
        "task": {
            "title": task_data["title"],
            "due_date": task_data["due_date"],
            "assigned_by": "coach",
        },
    })


@tool
def roadmap_generator(skills_and_goals: str) -> str:
    """
    Generate a structured weekly preparation roadmap for the student.
    Input: A description of the student's current skills, gaps, and target goals.
    Output: JSON roadmap with weekly plans.
    """
    if not _current_user_id:
        return "ERROR: No user context set."

    llm = get_agent_llm("coach", temperature=0.4)

    prompt = f"""Create a 4-week placement preparation roadmap based on this student profile.
Return ONLY valid JSON (no markdown) with this structure:
{{
    "total_weeks": 4,
    "weeks": [
        {{
            "week": 1,
            "theme": "Week theme",
            "goals": ["goal1", "goal2"],
            "tasks": [
                {{"title": "Task title", "type": "dsa|project|hr|technical", "estimated_hours": 5}}
            ],
            "milestones": ["milestone1"]
        }}
    ],
    "resources": ["resource1", "resource2"],
    "success_criteria": "How to know the student is ready"
}}

Student Profile:
{skills_and_goals}"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        roadmap_data = json.loads(content)

        # Save to database
        save_roadmap({
            "user_id": _current_user_id,
            "weeks": roadmap_data.get("weeks", []),
        })

        # Auto-create tasks from week 1
        week1 = roadmap_data.get("weeks", [{}])[0]
        for task_item in week1.get("tasks", [])[:3]:  # Max 3 auto-assigned tasks
            create_task({
                "user_id": _current_user_id,
                "title": task_item["title"],
                "due_date": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                "assigned_by": "coach",
            })

        logger.info(f"Roadmap generated for {_current_user_id}")
        return content

    except Exception as e:
        logger.error(f"Roadmap generation failed: {e}")
        return json.dumps({"error": str(e)})


@tool
def resource_recommender(skill_query: str) -> str:
    """
    Find learning resources for a specific skill or topic using the knowledge base (RAG).
    Input: The skill or topic to find resources for (e.g., "dynamic programming", "React hooks").
    Output: Curated list of recommended resources.
    """
    # Query RAG for learning resources
    resources = get_learning_resources(skill_query, top_k=3)
    dsa_resources = get_dsa_resources(skill_query, top_k=2)

    all_resources = resources + dsa_resources

    if not all_resources:
        return json.dumps({
            "skill": skill_query,
            "resources": [],
            "message": f"No specific resources found for '{skill_query}'. "
                       "Recommend checking LeetCode, freeCodeCamp, or Coursera.",
        })

    context = build_context_string(all_resources, max_chars=2000)
    return json.dumps({
        "skill": skill_query,
        "resources_found": len(all_resources),
        "content": context,
    })


# ────────────────────────────────────────────────────────────
#  Memory Management
# ────────────────────────────────────────────────────────────

def _load_memory(user_id: str) -> ConversationBufferWindowMemory:
    """Load conversation memory from Supabase."""
    memory = ConversationBufferWindowMemory(
        k=20,
        memory_key="chat_history",
        return_messages=True,
    )

    stored = get_coach_memory(user_id)
    for entry in stored:
        if entry.get("role") == "human":
            memory.chat_memory.add_user_message(entry["content"])
        elif entry.get("role") == "ai":
            memory.chat_memory.add_ai_message(entry["content"])

    return memory


def _save_memory(user_id: str, memory: ConversationBufferWindowMemory):
    """Persist conversation memory to Supabase."""
    messages = memory.chat_memory.messages
    serialized = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            serialized.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serialized.append({"role": "ai", "content": msg.content})

    # Keep only last 40 messages (k=20 pairs)
    serialized = serialized[-40:]
    update_coach_memory(user_id, serialized)


# ────────────────────────────────────────────────────────────
#  Agent Setup
# ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are NEXORA — a strict but supportive AI placement mentor.

YOUR PERSONALITY:
- You are NOT a generic chatbot. You are a decisive, experienced placement officer.
- You KNOW the student's complete history (skills, tasks, interview scores).
- You MAKE DECISIONS — you don't just suggest. You assign tasks, set deadlines, and escalate.
- You are encouraging but firm. If a student is slacking, you call it out directly.
- You speak with authority but genuine care.

YOUR CAPABILITIES:
1. task_assigner — Use this to create concrete tasks with deadlines for the student
2. roadmap_generator — Use this to create a structured weekly preparation plan
3. resource_recommender — Use this to find specific learning resources from the knowledge base

CRITICAL BEHAVIORS:
- If student mentions they missed tasks → IMMEDIATELY assign a recovery plan using task_assigner
- If student asks "what should I do?" → Generate a roadmap AND assign this week's tasks
- If student asks about a skill → Use resource_recommender to find relevant resources
- Always reference their skill profile when giving advice
- Never say "you could" or "you might want to" — say "Here's what you're doing this week:"

STUDENT CONTEXT:
{student_context}

OVERDUE TASKS:
{overdue_tasks}"""


def create_coach_agent(user_id: str) -> tuple:
    """Create the CoachAgent with memory and tools for a specific user."""
    _set_context(user_id)

    llm = get_agent_llm("coach", temperature=0.5, max_tokens=4096)
    tools = [task_assigner, roadmap_generator, resource_recommender]

    # Load persisted memory
    memory = _load_memory(user_id)

    # Build student context
    user = get_user(user_id) or {}
    skill_profile = user.get("skill_profile", {})
    readiness_score = user.get("readiness_score", 0)
    overdue = get_overdue_tasks(user_id)

    student_context = f"""
Name: {user.get('name', 'Student')}
Readiness Score: {readiness_score}/100
Skills: {json.dumps(skill_profile.get('found_skills', []))}
Missing Skills: {json.dumps(skill_profile.get('missing_skills', []))}
At Risk: {user.get('at_risk', False)}"""

    overdue_context = "None" if not overdue else json.dumps(
        [{"title": t["title"], "due_date": t["due_date"]} for t in overdue],
        indent=2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(
            student_context=student_context,
            overdue_tasks=overdue_context,
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    return executor, memory


# ────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────

async def chat_with_coach(user_id: str, message: str) -> dict:
    """
    Send a message to the CoachAgent and get a response.
    Memory is automatically loaded and saved.
    """
    logger.info(f"Coach chat for user {user_id}: {message[:100]}...")

    try:
        agent, memory = create_coach_agent(user_id)

        result = agent.invoke({"input": message})

        # Save updated memory
        _save_memory(user_id, memory)

        output = result.get("output", "")
        steps = result.get("intermediate_steps", [])

        # Extract any tool calls made
        tool_calls = []
        for step in steps:
            action, observation = step
            tool_calls.append({
                "tool": action.tool,
                "input": str(action.tool_input)[:200],
            })

        return {
            "success": True,
            "response": output,
            "tool_calls": tool_calls,
        }

    except Exception as e:
        logger.error(f"Coach chat failed for {user_id}: {e}")
        return {
            "success": False,
            "response": "I encountered an issue. Let me try again — what were you saying?",
            "error": str(e),
        }


async def get_conversation_history(user_id: str) -> list:
    """Fetch the full conversation history for a user."""
    return get_coach_memory(user_id)
