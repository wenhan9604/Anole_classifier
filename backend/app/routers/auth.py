from __future__ import annotations

import logging
from time import time
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from ..inat_settings import (
    INAT_SESSION_COOKIE_NAME,
    get_app_auth_flags,
    get_inat_oauth_settings,
    inat_oauth_configured,
    session_cookie_secure,
)
from ..services import inat_oauth_client
from ..services.inat_session_store import INatTokenRecord, store as inat_store

logger = logging.getLogger(__name__)

router = APIRouter()

SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

INAT_AUTHORIZE_BASE = "https://www.inaturalist.org/oauth/authorize"


class AuthResponse(BaseModel):
    accessToken: str
    refreshToken: str
    expiresAt: int


class INatStatusResponse(BaseModel):
    connected: bool
    expiresAt: Optional[int] = None


def _session_cookie_params() -> dict:
    s = get_inat_oauth_settings()
    same = (s.cookie_samesite or "lax").lower()
    if same not in ("lax", "strict", "none"):
        same = "lax"
    return {
        "httponly": True,
        "secure": session_cookie_secure(),
        "samesite": same,
        "max_age": SESSION_COOKIE_MAX_AGE,
        "path": "/",
    }


def _get_session_id(request: Request) -> Optional[str]:
    v = request.cookies.get(INAT_SESSION_COOKIE_NAME)
    return v if v else None


def _authorize_url(state: str) -> str:
    cfg = get_inat_oauth_settings()
    q: dict[str, str] = {
        "client_id": cfg.client_id,
        "redirect_uri": cfg.redirect_uri,
        "response_type": "code",
        "state": state,
    }
    if cfg.scopes.strip():
        q["scope"] = cfg.scopes.strip()
    return f"{INAT_AUTHORIZE_BASE}?{urlencode(q)}"


@router.post("/auth/mock-login", response_model=AuthResponse)
async def mock_login():
    if not get_app_auth_flags().enable_inat_mock_auth:
        raise HTTPException(status_code=404, detail="Mock auth disabled")
    now = int(time())
    return AuthResponse(
        accessToken="mock_access_token",
        refreshToken="mock_refresh_token",
        expiresAt=now + 3600,
    )


@router.get("/auth/inat/login")
async def inat_oauth_login(request: Request):
    if not inat_oauth_configured():
        raise HTTPException(
            status_code=503,
            detail="iNaturalist OAuth is not configured (set INAT_CLIENT_ID, INAT_CLIENT_SECRET, INAT_REDIRECT_URI, INAT_FRONTEND_SUCCESS_URL)",
        )
    session_id = _get_session_id(request) or inat_store.new_session_id()
    state = inat_store.new_oauth_state()
    inat_store.register_oauth_state(state, session_id)

    dest = _authorize_url(state)
    resp = RedirectResponse(url=dest, status_code=302)
    resp.set_cookie(INAT_SESSION_COOKIE_NAME, session_id, **_session_cookie_params())
    return resp


@router.get("/auth/inat/callback")
async def inat_oauth_callback(request: Request, code: Optional[str] = None, state: Optional[str] = None):
    cfg = get_inat_oauth_settings()
    if not inat_oauth_configured():
        raise HTTPException(status_code=503, detail="iNaturalist OAuth is not configured")

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state")

    session_id = inat_store.pop_oauth_session(state)
    if not session_id:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    try:
        raw = await inat_oauth_client.exchange_code_for_tokens(
            client_id=cfg.client_id,
            client_secret=cfg.client_secret,
            code=code,
            redirect_uri=cfg.redirect_uri,
        )
        access, refresh, expires_at = inat_oauth_client.parse_token_response(raw)
    except httpx.HTTPStatusError as e:
        logger.warning("iNat token exchange failed: %s %s", e.response.status_code, e.response.text)
        raise HTTPException(
            status_code=400,
            detail=f"iNaturalist token exchange failed: {e.response.text}",
        ) from e
    except (httpx.RequestError, ValueError) as e:
        logger.exception("iNat token exchange error")
        raise HTTPException(status_code=502, detail=str(e)) from e

    inat_store.set_tokens(session_id, INatTokenRecord(access_token=access, refresh_token=refresh, expires_at=expires_at))

    success = cfg.frontend_success_url.strip()
    if not success:
        raise HTTPException(status_code=500, detail="INAT_FRONTEND_SUCCESS_URL is not set")

    resp = RedirectResponse(url=success, status_code=302)
    resp.set_cookie(INAT_SESSION_COOKIE_NAME, session_id, **_session_cookie_params())
    return resp


@router.get("/auth/inat/status", response_model=INatStatusResponse)
async def inat_status(request: Request):
    sid = _get_session_id(request)
    if not sid:
        return INatStatusResponse(connected=False)
    rec = inat_store.get_tokens(sid)
    if not rec:
        return INatStatusResponse(connected=False)
    exp = int(rec.expires_at) if rec.expires_at > 0 else None
    return INatStatusResponse(connected=True, expiresAt=exp)


@router.post("/auth/inat/logout")
async def inat_logout(request: Request, response: Response):
    sid = _get_session_id(request)
    if sid:
        inat_store.clear_session(sid)
    response.delete_cookie(INAT_SESSION_COOKIE_NAME, path="/")
    return {"ok": True}
