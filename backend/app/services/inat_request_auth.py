"""Resolve iNaturalist Bearer token for the current browser session."""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx
from fastapi import Request

from ..inat_settings import INAT_SESSION_COOKIE_NAME, get_inat_oauth_settings
from . import inat_oauth_client
from .inat_session_store import INatTokenRecord, store as inat_store

logger = logging.getLogger(__name__)


def _bearer(access_token: str) -> str:
    token = access_token.strip()
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


async def resolve_inat_authorization_header(request: Request) -> Optional[str]:
    """
    Return Authorization header value for iNaturalist API calls, or None if not connected.
    Refreshes the access token when expired and a refresh_token is present.
    """
    sid = request.cookies.get(INAT_SESSION_COOKIE_NAME)
    if not sid:
        return None
    rec = inat_store.get_tokens(sid)
    if not rec:
        return None

    now = time.time()
    skew = 60.0
    if rec.expires_at > 0 and now >= rec.expires_at - skew:
        if rec.refresh_token:
            cfg = get_inat_oauth_settings()
            try:
                raw = await inat_oauth_client.refresh_access_token(
                    client_id=cfg.client_id,
                    client_secret=cfg.client_secret,
                    refresh_token=rec.refresh_token,
                )
                access, refresh, expires_at = inat_oauth_client.parse_token_response(raw)
                new_refresh = refresh or rec.refresh_token
                updated = INatTokenRecord(
                    access_token=access,
                    refresh_token=new_refresh,
                    expires_at=expires_at,
                )
                inat_store.set_tokens(sid, updated)
                rec = updated
            except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
                logger.warning("iNat token refresh failed: %s", e)
                return None
        else:
            return None

    return _bearer(rec.access_token)
