"""
NEXORA — AlertAgent
Risk-based alerting system triggered by scheduler / n8n webhooks.
Evaluates student risk and takes graduated actions via email and DB flags.

LLM: Groq (llama-3.1-70b) with Together AI fallback
Actions: Email (Resend), DB flags, Dashboard alerts
Severity: LOW → MEDIUM → HIGH → CRITICAL
"""

import json
import logging
from datetime import datetime, timezone

import resend

from chains.orchestrator import get_agent_llm
from db.supabase_client import (
    get_user,
    get_tasks,
    get_overdue_tasks,
    get_user_interview_sessions,
    create_alert,
    flag_at_risk,
)
from config import get_settings

logger = logging.getLogger("nexora.agents.alert")


# ────────────────────────────────────────────────────────────
#  Risk Evaluation
# ────────────────────────────────────────────────────────────

TRIGGER_REASONS = {
    "missed_tasks": "Student has missed assigned task deadlines",
    "low_score": "Student scored below threshold in mock interviews",
    "deadline_approaching": "Placement deadline is approaching with low readiness",
    "inactivity": "Student has been inactive for extended period",
    "critical_gaps": "Critical skill gaps detected in latest assessment",
}


def evaluate_risk(user_id: str, trigger_reason: str) -> dict:
    """
    Evaluate the risk level for a student based on trigger reason
    and their current status.
    
    Returns: { severity, risk_factors, summary }
    """
    user = get_user(user_id)
    if not user:
        return {"severity": "LOW", "risk_factors": [], "summary": "User not found"}

    risk_factors = []
    severity_score = 0

    # Check overdue tasks
    overdue = get_overdue_tasks(user_id)
    if overdue:
        risk_factors.append(f"{len(overdue)} overdue task(s)")
        severity_score += min(len(overdue) * 15, 40)

    # Check readiness score
    readiness = user.get("readiness_score", 0)
    if readiness < 30:
        risk_factors.append(f"Very low readiness score: {readiness}/100")
        severity_score += 30
    elif readiness < 50:
        risk_factors.append(f"Below-average readiness: {readiness}/100")
        severity_score += 15

    # Check interview performance
    sessions = get_user_interview_sessions(user_id)
    if sessions:
        recent_scores = []
        for s in sessions[:3]:
            scores = s.get("scores", {})
            avg = scores.get("running_average", 0)
            if avg > 0:
                recent_scores.append(avg)
        if recent_scores:
            avg_score = sum(recent_scores) / len(recent_scores)
            if avg_score < 4:
                risk_factors.append(f"Poor interview average: {avg_score:.1f}/10")
                severity_score += 25
            elif avg_score < 6:
                risk_factors.append(f"Below-par interview average: {avg_score:.1f}/10")
                severity_score += 10

    # Check if already at risk
    if user.get("at_risk"):
        risk_factors.append("Already flagged as at-risk")
        severity_score += 10

    # Add trigger-specific factors
    trigger_desc = TRIGGER_REASONS.get(trigger_reason, trigger_reason)
    risk_factors.append(f"Trigger: {trigger_desc}")

    if trigger_reason == "missed_tasks":
        severity_score += 20
    elif trigger_reason == "low_score":
        severity_score += 15
    elif trigger_reason == "deadline_approaching":
        severity_score += 25
    elif trigger_reason == "critical_gaps":
        severity_score += 30

    # Determine severity level
    if severity_score >= 70:
        severity = "CRITICAL"
    elif severity_score >= 50:
        severity = "HIGH"
    elif severity_score >= 30:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "severity": severity,
        "severity_score": severity_score,
        "risk_factors": risk_factors,
        "readiness_score": readiness,
        "overdue_tasks": len(overdue),
        "summary": f"Risk assessment for {user.get('name', 'Student')}: "
                   f"{severity} ({severity_score}/100) — {', '.join(risk_factors[:3])}",
    }


# ────────────────────────────────────────────────────────────
#  Email Actions (Resend)
# ────────────────────────────────────────────────────────────

def _init_resend():
    """Initialize Resend with API key."""
    settings = get_settings()
    resend.api_key = settings.RESEND_API_KEY


def send_student_nudge(user: dict, risk: dict) -> bool:
    """Send a motivational nudge email to the student."""
    _init_resend()
    settings = get_settings()

    # Generate personalized message
    llm = get_agent_llm("alert", temperature=0.7, max_tokens=1024)
    prompt = f"""Write a short, encouraging email nudge for a placement student.

Student name: {user.get('name', 'Student')}
Issue: {risk.get('summary', 'Needs attention')}
Risk factors: {json.dumps(risk.get('risk_factors', []))}

Rules:
- Keep it under 150 words
- Be encouraging but honest
- Include one specific actionable step
- Sign off as "NEXORA — Your AI Placement Coach"
- Don't be preachy or generic"""

    try:
        response = llm.invoke(prompt)
        email_body = response.content.strip()
    except Exception:
        email_body = (
            f"Hi {user.get('name', 'there')},\n\n"
            f"We noticed you might need some support with your placement preparation. "
            f"{''.join(risk.get('risk_factors', ['Keep pushing!'])[:2])}\n\n"
            f"Log in to NEXORA and let's get back on track!\n\n"
            f"— NEXORA, Your AI Placement Coach"
        )

    try:
        resend.Emails.send({
            "from": settings.FROM_EMAIL,
            "to": [user.get("email", "")],
            "subject": "📌 NEXORA — Let's get back on track!",
            "html": f"<div style='font-family: Inter, sans-serif; line-height: 1.6;'>"
                    f"<pre style='white-space: pre-wrap;'>{email_body}</pre></div>",
        })
        logger.info(f"Nudge email sent to {user.get('email')}")
        return True
    except Exception as e:
        logger.error(f"Failed to send nudge email: {e}")
        return False


def send_tpc_alert(user: dict, risk: dict) -> bool:
    """Send alert email to TPC (Training & Placement Cell) about a student."""
    _init_resend()
    settings = get_settings()

    email_body = f"""
    <div style="font-family: Inter, sans-serif; background: #0e1322; color: #dee1f7; padding: 24px; border-radius: 16px;">
        <h2 style="color: #c0c1ff;">⚠️ NEXORA — Student Alert</h2>
        <hr style="border-color: rgba(99,102,241,0.2);">
        
        <p><strong>Student:</strong> {user.get('name', 'Unknown')} ({user.get('email', '')})</p>
        <p><strong>Severity:</strong> <span style="color: {'#ff4444' if risk['severity'] in ('HIGH', 'CRITICAL') else '#ffb783'};">
            {risk['severity']}
        </span></p>
        <p><strong>Readiness Score:</strong> {risk.get('readiness_score', 'N/A')}/100</p>
        <p><strong>Overdue Tasks:</strong> {risk.get('overdue_tasks', 0)}</p>
        
        <h3 style="color: #c0c1ff;">Risk Factors</h3>
        <ul>
            {''.join(f'<li>{f}</li>' for f in risk.get('risk_factors', []))}
        </ul>
        
        <h3 style="color: #c0c1ff;">Recommended Action</h3>
        <p>{risk.get('summary', 'Review student profile and schedule a meeting.')}</p>
        
        <hr style="border-color: rgba(99,102,241,0.2);">
        <p style="color: #908fa0; font-size: 12px;">
            Sent by NEXORA AI Alert System at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
        </p>
    </div>
    """

    try:
        # Send to TPC — in production, this would be a configured TPC email
        resend.Emails.send({
            "from": settings.FROM_EMAIL,
            "to": [settings.FROM_EMAIL],  # Replace with actual TPC email
            "subject": f"🚨 NEXORA Alert [{risk['severity']}] — {user.get('name', 'Student')}",
            "html": email_body,
        })
        logger.info(f"TPC alert sent for {user.get('name')} ({risk['severity']})")
        return True
    except Exception as e:
        logger.error(f"Failed to send TPC alert: {e}")
        return False


# ────────────────────────────────────────────────────────────
#  Alert Actions (graduated response)
# ────────────────────────────────────────────────────────────

async def process_alert(user_id: str, trigger_reason: str) -> dict:
    """
    Full alert processing pipeline:
    1. Evaluate risk level
    2. Take graduated actions based on severity
    3. Log the alert to DB
    """
    logger.info(f"Processing alert for {user_id}: {trigger_reason}")

    # 1. Evaluate risk
    risk = evaluate_risk(user_id, trigger_reason)
    severity = risk["severity"]
    user = get_user(user_id)

    if not user:
        return {"success": False, "error": "User not found"}

    actions_taken = []

    # 2. Take graduated actions
    if severity == "LOW":
        # Send student nudge email
        if send_student_nudge(user, risk):
            actions_taken.append("student_nudge_email")

    elif severity == "MEDIUM":
        # Send nudge + update dashboard alert
        if send_student_nudge(user, risk):
            actions_taken.append("student_nudge_email")
        actions_taken.append("dashboard_alert_updated")

    elif severity == "HIGH":
        # Send TPC email + student nudge
        if send_student_nudge(user, risk):
            actions_taken.append("student_nudge_email")
        if send_tpc_alert(user, risk):
            actions_taken.append("tpc_alert_email")

    elif severity == "CRITICAL":
        # Full escalation: all emails + flag at-risk
        if send_student_nudge(user, risk):
            actions_taken.append("student_nudge_email")
        if send_tpc_alert(user, risk):
            actions_taken.append("tpc_alert_email")
        flag_at_risk(user_id, True)
        actions_taken.append("flagged_at_risk")

    # 3. Log alert to DB
    alert_record = create_alert({
        "user_id": user_id,
        "type": trigger_reason,
        "severity": severity,
        "message": risk["summary"],
    })

    result = {
        "success": True,
        "user_id": user_id,
        "severity": severity,
        "risk_assessment": risk,
        "actions_taken": actions_taken,
        "alert_id": alert_record.get("id"),
    }

    logger.info(f"Alert processed: {severity} — actions: {actions_taken}")
    return result
