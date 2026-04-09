"""
NEXORA — Supabase Authentication
Email + Password signup/login flow + JWT validation + auto-profile creation.
"""

import logging
from typing import Optional

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from db.supabase_client import get_anon_supabase, get_supabase, get_user, upsert_user
from config import get_settings

logger = logging.getLogger("nexora.auth")

security = HTTPBearer(auto_error=False)


def _auth_email_redirect_url() -> str:
    """URL Supabase uses in confirmation emails — must match Dashboard redirect allowlist."""
    settings = get_settings()
    if settings.AUTH_EMAIL_REDIRECT_URL:
        return settings.AUTH_EMAIL_REDIRECT_URL.rstrip("/")
    base = settings.NEXT_PUBLIC_APP_URL.rstrip("/")
    return f"{base}/auth/callback"


def get_auth_email_redirect_url() -> str:
    """Public helper for health/status endpoints and docs."""
    return _auth_email_redirect_url()


# ────────────────────────────────────────────────────────────
#  Signup + Login (Email + Password)
# ────────────────────────────────────────────────────────────

async def signup_user(email: str, password: str, name: str) -> dict:
    """
    Register a new user with email + password.
    Also creates a profile row in the users table.
    """
    try:
        client = get_anon_supabase()
        redirect_to = _auth_email_redirect_url()
        response = client.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {
                    # Stored on the auth user and visible in Supabase → Authentication → Users
                    "data": {
                        "name": name,
                        "full_name": name,
                    },
                    # Critical: confirmation email link must redirect here (add URL in Supabase Dashboard)
                    "email_redirect_to": redirect_to,
                },
            }
        )

        user = response.user
        session = response.session

        if not user:
            raise HTTPException(status_code=400, detail="Signup failed — try a different email")

        # Create profile in users table (using service-role client)
        await _ensure_user_profile(user.id, email, name)

        # If email confirmation is enabled, session may be None
        if session:
            logger.info(f"User {email} signed up and auto-logged in")
            return {
                "success": True,
                "message": "Account created successfully",
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
                "auth_redirect_url": redirect_to,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "name": name,
                },
            }
        else:
            # Email confirmation required but we'll auto-confirm via admin API
            try:
                admin_client = get_supabase()
                admin_client.auth.admin.update_user_by_id(
                    user.id,
                    {
                        "email_confirm": True,
                        "user_metadata": {"name": name, "full_name": name},
                    },
                )
                # Now log them in
                login_response = client.auth.sign_in_with_password({
                    "email": email,
                    "password": password,
                })
                sess = login_response.session
                if sess:
                    logger.info(f"User {email} signed up, auto-confirmed, and logged in")
                    return {
                        "success": True,
                        "message": "Account created successfully",
                        "access_token": sess.access_token,
                        "refresh_token": sess.refresh_token,
                        "auth_redirect_url": redirect_to,
                        "user": {
                            "id": user.id,
                            "email": user.email,
                            "name": name,
                        },
                    }
            except Exception as confirm_err:
                logger.warning(f"Auto-confirm failed: {confirm_err}")

            return {
                "success": True,
                "message": (
                    "Account created. Supabase sent a confirmation link (not an SMS OTP). "
                    "Open the email and click the link — it returns you to this app to finish sign-in."
                ),
                "needs_confirmation": True,
                "auth_redirect_url": redirect_to,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "name": name,
                },
            }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "already registered" in error_msg.lower() or "already been registered" in error_msg.lower():
            raise HTTPException(status_code=409, detail="This email is already registered. Please login instead.")
        logger.error(f"Signup failed for {email}: {e}")
        raise HTTPException(status_code=400, detail=f"Signup failed: {error_msg}")


async def login_user(email: str, password: str) -> dict:
    """
    Log in with email + password. Returns JWT session.
    """
    try:
        client = get_anon_supabase()
        response = client.auth.sign_in_with_password({
            "email": email,
            "password": password,
        })

        session = response.session
        user = response.user

        if not session or not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Ensure profile exists
        profile = await _ensure_user_profile(user.id, email)

        logger.info(f"User {email} logged in successfully")
        return {
            "success": True,
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": profile.get("name", email.split("@")[0]),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "invalid" in error_msg.lower() or "credentials" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid email or password")
        logger.error(f"Login failed for {email}: {e}")
        raise HTTPException(status_code=401, detail=f"Login failed: {error_msg}")


# ────────────────────────────────────────────────────────────
#  Profile Auto-Creation
# ────────────────────────────────────────────────────────────

async def _ensure_user_profile(user_id: str, email: str, name: str = None) -> dict:
    """Create a user profile row if it doesn't exist yet."""
    existing = get_user(user_id)
    if existing:
        # Update name if provided and currently empty
        if name and (not existing.get("name") or existing["name"] == email.split("@")[0]):
            from db.supabase_client import update_user_field
            update_user_field(user_id, "name", name)
            existing["name"] = name
        return existing

    logger.info(f"Creating new profile for {email} (id={user_id})")
    profile = {
        "id": user_id,
        "email": email,
        "name": name or email.split("@")[0],
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
