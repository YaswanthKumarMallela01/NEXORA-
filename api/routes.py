"""
NEXORA — API Routes
All endpoint definitions, grouped by feature domain.

Routers:
  /auth/*           — OTP authentication
  /api/resume/*     — Resume analysis
  /api/coach/*      — CoachAgent chat
  /api/interview/*  — Mock interview sessions
  /api/dashboard/*  — Student + TPC dashboards
  /api/alerts/*     — n8n webhook triggers
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from auth.supabase_auth import (
    send_otp,
    verify_otp,
    get_current_user,
    require_tpc_role,
)
from agents.resume_agent import analyze_resume
from agents.coach_agent import chat_with_coach, get_conversation_history
from agents.interview_agent import start_interview, submit_answer, get_session_summary
from agents.alert_agent import process_alert
from api.middleware import validate_webhook_key
from db.supabase_client import (
    get_student_dashboard,
    get_tpc_dashboard,
    get_tasks,
    update_task,
    acknowledge_alert,
)

logger = logging.getLogger("nexora.api.routes")


# ════════════════════════════════════════════════════════════
#  Request/Response Models
# ════════════════════════════════════════════════════════════

# ── Auth ──
class SendOTPRequest(BaseModel):
    email: EmailStr


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    token: str


# ── Resume ──
class ResumeAnalyzeRequest(BaseModel):
    pdf_base64: str = Field(..., description="Base64-encoded PDF file content")


# ── Coach ──
class CoachChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


# ── Interview ──
class InterviewStartRequest(BaseModel):
    role: str = Field(..., description="Target role (e.g., 'Software Engineer')")


class InterviewAnswerRequest(BaseModel):
    session_id: str
    answer: str = Field(..., min_length=1, max_length=10000)


# ── Alerts ──
class AlertTriggerRequest(BaseModel):
    user_id: str
    trigger_reason: str = Field(
        ...,
        description="One of: missed_tasks, low_score, deadline_approaching, inactivity, critical_gaps",
    )


# ── Tasks ──
class TaskUpdateRequest(BaseModel):
    status: Optional[str] = None
    due_date: Optional[str] = None


# ════════════════════════════════════════════════════════════
#  AUTH ROUTER
# ════════════════════════════════════════════════════════════

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/send-otp")
async def route_send_otp(req: SendOTPRequest):
    """Send OTP to user's email via Supabase Magic Link."""
    return await send_otp(req.email)


@auth_router.post("/verify-otp")
async def route_verify_otp(req: VerifyOTPRequest):
    """Verify OTP and return JWT session."""
    return await verify_otp(req.email, req.token)


# ════════════════════════════════════════════════════════════
#  RESUME ROUTER
# ════════════════════════════════════════════════════════════

resume_router = APIRouter(prefix="/api/resume", tags=["Resume Analysis"])


@resume_router.post("/analyze")
async def route_analyze_resume(
    req: ResumeAnalyzeRequest,
    user: dict = Depends(get_current_user),
):
    """
    Upload and analyze a resume PDF.
    Runs the full ResumeAgent pipeline:
    pdf_parser → skill_extractor → jd_matcher → readiness score.
    """
    result = await analyze_resume(user["id"], req.pdf_base64)

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

    return result


# ════════════════════════════════════════════════════════════
#  COACH ROUTER
# ════════════════════════════════════════════════════════════

coach_router = APIRouter(prefix="/api/coach", tags=["Coach Agent"])


@coach_router.post("/chat")
async def route_coach_chat(
    req: CoachChatRequest,
    user: dict = Depends(get_current_user),
):
    """
    Send a message to the CoachAgent.
    The agent has conversation memory and can assign tasks,
    generate roadmaps, and recommend resources.
    """
    result = await chat_with_coach(user["id"], req.message)

    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Coach interaction failed"),
        )

    return result


@coach_router.get("/memory")
async def route_get_memory(user: dict = Depends(get_current_user)):
    """Fetch the full conversation history with the CoachAgent."""
    history = await get_conversation_history(user["id"])
    return {"success": True, "history": history, "message_count": len(history)}


# ════════════════════════════════════════════════════════════
#  INTERVIEW ROUTER
# ════════════════════════════════════════════════════════════

interview_router = APIRouter(prefix="/api/interview", tags=["Mock Interview"])


@interview_router.post("/start")
async def route_start_interview(
    req: InterviewStartRequest,
    user: dict = Depends(get_current_user),
):
    """
    Begin a new mock interview session.
    Creates a session and generates the first question.
    """
    result = await start_interview(user["id"], req.role)

    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Failed to start interview"),
        )

    return result


@interview_router.post("/answer")
async def route_submit_answer(
    req: InterviewAnswerRequest,
    user: dict = Depends(get_current_user),
):
    """
    Submit an answer to the current interview question.
    Returns evaluation feedback and the next question (or session summary).
    """
    result = await submit_answer(user["id"], req.session_id, req.answer)

    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Answer processing failed"),
        )

    return result


@interview_router.get("/summary/{session_id}")
async def route_get_summary(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    """Get the full summary and scores of a completed interview session."""
    result = await get_session_summary(session_id)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Session not found"))

    return result


# ════════════════════════════════════════════════════════════
#  DASHBOARD ROUTER
# ════════════════════════════════════════════════════════════

dashboard_router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@dashboard_router.get("/student")
async def route_student_dashboard(user: dict = Depends(get_current_user)):
    """
    Full student dashboard data.
    Includes: profile, skills, readiness score, tasks, interviews, alerts, roadmap.
    """
    data = get_student_dashboard(user["id"])
    return {"success": True, **data}


@dashboard_router.get("/tpc")
async def route_tpc_dashboard(user: dict = Depends(require_tpc_role)):
    """
    TPC admin dashboard.
    Includes: all students, at-risk students, recent alerts.
    Requires TPC role.
    """
    data = get_tpc_dashboard()
    return {"success": True, **data}


# ════════════════════════════════════════════════════════════
#  ALERTS ROUTER
# ════════════════════════════════════════════════════════════

alerts_router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@alerts_router.post("/trigger")
async def route_trigger_alert(
    req: AlertTriggerRequest,
    request: Request,
):
    """
    Trigger an alert evaluation and action pipeline.
    Called by n8n webhooks or internal schedulers.
    Validates webhook API key.
    """
    await validate_webhook_key(request)

    valid_reasons = {"missed_tasks", "low_score", "deadline_approaching", "inactivity", "critical_gaps"}
    if req.trigger_reason not in valid_reasons:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid trigger_reason. Must be one of: {valid_reasons}",
        )

    result = await process_alert(req.user_id, req.trigger_reason)

    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Alert processing failed"),
        )

    return result


@alerts_router.post("/acknowledge/{alert_id}")
async def route_acknowledge_alert(
    alert_id: str,
    user: dict = Depends(get_current_user),
):
    """Mark an alert as acknowledged."""
    result = acknowledge_alert(alert_id)
    return {"success": True, "alert": result}


# ════════════════════════════════════════════════════════════
#  TASKS ROUTER (bonus — helpful for frontend)
# ════════════════════════════════════════════════════════════

tasks_router = APIRouter(prefix="/api/tasks", tags=["Tasks"])


@tasks_router.get("/")
async def route_get_tasks(
    status: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    """Fetch all tasks for the current user, optionally filtered by status."""
    tasks = get_tasks(user["id"], status=status)
    return {"success": True, "tasks": tasks, "count": len(tasks)}


@tasks_router.patch("/{task_id}")
async def route_update_task(
    task_id: str,
    req: TaskUpdateRequest,
    user: dict = Depends(get_current_user),
):
    """Update a task's status or due date."""
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    result = update_task(task_id, updates)
    return {"success": True, "task": result}
