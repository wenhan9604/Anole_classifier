"""Exchange authorization codes and refresh tokens with iNaturalist."""

from __future__ import annotations

import time
from typing import Any

import httpx

INAT_TOKEN_URL = "https://www.inaturalist.org/oauth/token"


async def exchange_code_for_tokens(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            INAT_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()


async def refresh_access_token(
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            INAT_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={"Accept": "application/json"},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()


def parse_token_response(data: dict[str, Any]) -> tuple[str, str, float]:
    """Returns (access_token, refresh_token, expires_at_unix)."""
    access = data.get("access_token")
    if not access or not isinstance(access, str):
        raise ValueError("Token response missing access_token")
    refresh = data.get("refresh_token") or ""
    if not isinstance(refresh, str):
        refresh = ""
    expires_in = data.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = time.time() + float(expires_in)
    else:
        expires_at = 0.0
    return access, refresh, expires_at
