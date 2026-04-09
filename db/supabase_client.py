"""
NEXORA — Supabase Database Client
Full CRUD helpers for all tables: users, tasks, interview_sessions, alerts, roadmap.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from supabase import create_client, Client

from config import get_settings

logger = logging.getLogger("nexora.db")

# ────────────────────────────────────────────────────────────
#  Client Initialization
# ────────────────────────────────────────────────────────────

_client: Optional[Client] = None


def get_supabase() -> Client:
    """Return a cached Supabase client (service-role for backend ops)."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        logger.info("Supabase client initialized (service-role)")
    return _client


def get_anon_supabase() -> Client:
    """Return an anon-key client for auth operations."""
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)


# ────────────────────────────────────────────────────────────
#  USERS
# ────────────────────────────────────────────────────────────

def get_user(user_id: str) -> Optional[dict]:
    """Fetch a user by their Supabase auth UID."""
    try:
        resp = get_supabase().table("users").select("*").eq("id", user_id).limit(1).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        logger.warning(f"get_user failed for {user_id}: {e}")
        return None


def get_user_by_email(email: str) -> Optional[dict]:
    """Fetch a user by email."""
    try:
        resp = get_supabase().table("users").select("*").eq("email", email).limit(1).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        logger.warning(f"get_user_by_email failed for {email}: {e}")
        return None


def upsert_user(data: dict) -> dict:
    """Insert or update a user profile."""
    resp = get_supabase().table("users").upsert(data, on_conflict="id").execute()
    return resp.data[0] if resp.data else {}


def update_user_field(user_id: str, field: str, value: Any) -> dict:
    """Update a single field on a user record."""
    resp = get_supabase().table("users").update({field: value}).eq("id", user_id).execute()
    return resp.data[0] if resp.data else {}


def update_skill_profile(user_id: str, skill_profile: dict) -> dict:
    """Save the ResumeAgent output to users.skill_profile."""
    return update_user_field(user_id, "skill_profile", skill_profile)


def update_readiness_score(user_id: str, score: int) -> dict:
    """Update readiness score (0–100)."""
    return update_user_field(user_id, "readiness_score", score)


def flag_at_risk(user_id: str, at_risk: bool = True) -> dict:
    """Flag/unflag a student as at-risk."""
    return update_user_field(user_id, "at_risk", at_risk)


# ────────────────────────────────────────────────────────────
#  COACH MEMORY
# ────────────────────────────────────────────────────────────

def get_coach_memory(user_id: str) -> list:
    """Load conversation memory JSON from users.coach_memory."""
    user = get_user(user_id)
    if user and user.get("coach_memory"):
        mem = user["coach_memory"]
        return mem if isinstance(mem, list) else json.loads(mem)
    return []


def update_coach_memory(user_id: str, memory_json: list) -> dict:
    """Save conversation memory to users.coach_memory."""
    return update_user_field(user_id, "coach_memory", memory_json)


# ────────────────────────────────────────────────────────────
#  TASKS
# ────────────────────────────────────────────────────────────

def get_tasks(user_id: str, status: Optional[str] = None) -> list:
    """Fetch tasks for a user, optionally filtered by status."""
    query = get_supabase().table("tasks").select("*").eq("user_id", user_id)
    if status:
        query = query.eq("status", status)
    resp = query.order("created_at", desc=True).execute()
    return resp.data or []


def create_task(data: dict) -> dict:
    """Create a new task. Expected keys: user_id, title, due_date, assigned_by."""
    task = {
        "user_id": data["user_id"],
        "title": data["title"],
        "due_date": data.get("due_date"),
        "status": data.get("status", "pending"),
        "assigned_by": data.get("assigned_by", "agent"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    resp = get_supabase().table("tasks").insert(task).execute()
    return resp.data[0] if resp.data else {}


def update_task(task_id: str, updates: dict) -> dict:
    """Update task fields (e.g. status, due_date)."""
    resp = get_supabase().table("tasks").update(updates).eq("id", task_id).execute()
    return resp.data[0] if resp.data else {}


def get_overdue_tasks(user_id: str) -> list:
    """Fetch overdue pending tasks for a user."""
    now = datetime.now(timezone.utc).isoformat()
    resp = (
        get_supabase()
        .table("tasks")
        .select("*")
        .eq("user_id", user_id)
        .eq("status", "pending")
        .lt("due_date", now)
        .execute()
    )
    return resp.data or []


# ────────────────────────────────────────────────────────────
#  INTERVIEW SESSIONS
# ────────────────────────────────────────────────────────────

def create_interview_session(data: dict) -> dict:
    """Create a new interview session."""
    session = {
        "user_id": data["user_id"],
        "role": data["role"],
        "questions": data.get("questions", []),
        "answers": data.get("answers", []),
        "scores": data.get("scores", {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    resp = get_supabase().table("interview_sessions").insert(session).execute()
    return resp.data[0] if resp.data else {}


def get_interview_session(session_id: str) -> Optional[dict]:
    """Fetch a specific interview session."""
    try:
        resp = (
            get_supabase()
            .table("interview_sessions")
            .select("*")
            .eq("id", session_id)
            .limit(1)
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception as e:
        logger.warning(f"get_interview_session failed: {e}")
        return None


def update_interview_session(session_id: str, updates: dict) -> dict:
    """Update interview session (add answers, scores, etc.)."""
    resp = (
        get_supabase()
        .table("interview_sessions")
        .update(updates)
        .eq("id", session_id)
        .execute()
    )
    return resp.data[0] if resp.data else {}


def get_user_interview_sessions(user_id: str) -> list:
    """Fetch all interview sessions for a user."""
    resp = (
        get_supabase()
        .table("interview_sessions")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []


# ────────────────────────────────────────────────────────────
#  ALERTS
# ────────────────────────────────────────────────────────────

def create_alert(data: dict) -> dict:
    """Log an alert to the alerts table."""
    alert = {
        "user_id": data["user_id"],
        "type": data["type"],
        "severity": data["severity"],
        "message": data["message"],
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "acknowledged": False,
    }
    resp = get_supabase().table("alerts").insert(alert).execute()
    return resp.data[0] if resp.data else {}


def get_alerts(user_id: str, acknowledged: Optional[bool] = None) -> list:
    """Fetch alerts for a user, optionally filtered by acknowledged status."""
    query = get_supabase().table("alerts").select("*").eq("user_id", user_id)
    if acknowledged is not None:
        query = query.eq("acknowledged", acknowledged)
    resp = query.order("sent_at", desc=True).execute()
    return resp.data or []


def acknowledge_alert(alert_id: str) -> dict:
    """Mark an alert as acknowledged."""
    resp = (
        get_supabase()
        .table("alerts")
        .update({"acknowledged": True})
        .eq("id", alert_id)
        .execute()
    )
    return resp.data[0] if resp.data else {}


# ────────────────────────────────────────────────────────────
#  ROADMAP
# ────────────────────────────────────────────────────────────

def save_roadmap(data: dict) -> dict:
    """Save a generated roadmap for a user."""
    roadmap = {
        "user_id": data["user_id"],
        "weeks": data["weeks"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    resp = get_supabase().table("roadmap").insert(roadmap).execute()
    return resp.data[0] if resp.data else {}


def get_roadmap(user_id: str) -> Optional[dict]:
    """Get the latest roadmap for a user."""
    try:
        resp = (
            get_supabase()
            .table("roadmap")
            .select("*")
            .eq("user_id", user_id)
            .order("generated_at", desc=True)
            .limit(1)
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception as e:
        logger.warning(f"get_roadmap failed for {user_id}: {e}")
        return None


# ────────────────────────────────────────────────────────────
#  DASHBOARD AGGREGATIONS
# ────────────────────────────────────────────────────────────

def get_student_dashboard(user_id: str) -> dict:
    """Aggregate all data for the student dashboard."""
    user = get_user(user_id) or {}

    try:
        tasks = get_tasks(user_id)
    except Exception as e:
        logger.warning(f"Dashboard: tasks query failed: {e}")
        tasks = []

    try:
        sessions = get_user_interview_sessions(user_id)
    except Exception as e:
        logger.warning(f"Dashboard: interviews query failed: {e}")
        sessions = []

    try:
        alerts = get_alerts(user_id, acknowledged=False)
    except Exception as e:
        logger.warning(f"Dashboard: alerts query failed: {e}")
        alerts = []

    try:
        roadmap = get_roadmap(user_id)
    except Exception as e:
        logger.warning(f"Dashboard: roadmap query failed: {e}")
        roadmap = None

    pending = [t for t in tasks if t.get("status") == "pending"]
    completed = [t for t in tasks if t.get("status") == "completed"]

    return {
        "profile": user,
        "skill_profile": user.get("skill_profile", {}),
        "readiness_score": user.get("readiness_score", 0),
        "at_risk": user.get("at_risk", False),
        "tasks": {
            "total": len(tasks),
            "pending": len(pending),
            "completed": len(completed),
            "items": tasks[:10],
        },
        "interviews": {
            "total": len(sessions),
            "recent": sessions[:5],
        },
        "active_alerts": alerts[:5],
        "roadmap": roadmap,
    }


def get_tpc_dashboard() -> dict:
    """Aggregate all student data for TPC admin dashboard."""
    resp = get_supabase().table("users").select("*").eq("role", "student").execute()
    students = resp.data or []

    at_risk = [s for s in students if s.get("at_risk")]

    # Get recent unacknowledged alerts across all users
    alerts_resp = (
        get_supabase()
        .table("alerts")
        .select("*")
        .eq("acknowledged", False)
        .order("sent_at", desc=True)
        .limit(20)
        .execute()
    )

    return {
        "total_students": len(students),
        "at_risk_count": len(at_risk),
        "at_risk_students": at_risk,
        "students": students,
        "recent_alerts": alerts_resp.data or [],
    }
