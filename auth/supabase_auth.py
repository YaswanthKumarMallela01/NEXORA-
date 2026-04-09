"""
NEXORA — Supabase Authentication
Magic-link OTP flow + JWT validation + auto-profile creation.
"""

import logging
from typing import Optional

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from db.supabase_client import get_anon_supabase, get_supabase, get_user, upsert_user
from config import get_settings

logger = logging.getLogger("nexora.auth")

security = HTTPBearer(auto_error=False)


# ────────────────────────────────────────────────────────────
#  OTP Flow
# ────────────────────────────────────────────────────────────

async def send_otp(email: str) -> dict:
    """
    Send OTP to user's email via Supabase Magic Link.
    Uses the anon client so Supabase handles the email delivery.
    """
    try:
        client = get_anon_supabase()
        response = client.auth.sign_in_with_otp({"email": email})
        logger.info(f"OTP sent to {email}")
        return {"success": True, "message": f"OTP sent to {email}"}
    except Exception as e:
        logger.error(f"Failed to send OTP to {email}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to send OTP: {str(e)}")


async def verify_otp(email: str, token: str) -> dict:
    """
    Verify OTP token and return session.
    On first login, auto-creates user profile in the users table.
    """
    try:
        client = get_anon_supabase()
        response = client.auth.verify_otp({
            "email": email,
            "token": token,
            "type": "email",
        })

        session = response.session
        user = response.user

        if not session or not user:
            raise HTTPException(status_code=401, detail="Invalid or expired OTP")

        # Auto-create profile on first login
        await _ensure_user_profile(user.id, email)

        logger.info(f"User {email} verified successfully")
        return {
            "success": True,
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
            "user": {
                "id": user.id,
                "email": user.email,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OTP verification failed for {email}: {e}")
        raise HTTPException(status_code=401, detail=f"Verification failed: {str(e)}")


# ────────────────────────────────────────────────────────────
#  Profile Auto-Creation
# ────────────────────────────────────────────────────────────

async def _ensure_user_profile(user_id: str, email: str) -> dict:
    """Create a user profile row if it doesn't exist yet."""
    existing = get_user(user_id)
    if existing:
        return existing

    logger.info(f"Creating new profile for {email} (id={user_id})")
    profile = {
        "id": user_id,
        "email": email,
        "name": email.split("@")[0],
        "role": "student",
        "skill_profile": {},
        "coach_memory": [],
        "readiness_score": 0,
        "at_risk": False,
    }
    return upsert_user(profile)


# ────────────────────────────────────────────────────────────
#  JWT Validation Dependency
# ────────────────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    FastAPI dependency — validates JWT from Authorization header.
    Returns the authenticated user's profile from the DB.
    """
    token = None

    # Try Bearer token from header
    if credentials:
        token = credentials.credentials

    # Fallback: try cookie
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated — provide Bearer token or access_token cookie",
        )

    try:
        # Validate token via Supabase
        client = get_anon_supabase()
        user_response = client.auth.get_user(token)
        auth_user = user_response.user

        if not auth_user:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Fetch full profile from DB
        profile = get_user(auth_user.id)
        if not profile:
            # Edge case: auth exists but profile doesn't — create it
            profile = await _ensure_user_profile(auth_user.id, auth_user.email)

        return profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(status_code=401, detail="Token validation failed")


async def require_tpc_role(user: dict = Depends(get_current_user)) -> dict:
    """Dependency that requires TPC (admin) role."""
    if user.get("role") != "tpc":
        raise HTTPException(status_code=403, detail="TPC admin access required")
    return user
