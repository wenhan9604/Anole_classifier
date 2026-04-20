"""In-memory iNaturalist OAuth state and per-session tokens (swap for Redis/DB later)."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class INatTokenRecord:
    access_token: str
    refresh_token: str
    expires_at: float  # unix timestamp; 0 if unknown


class INatSessionStore:
    """Thread-safe store for OAuth CSRF state and session-bound iNat tokens."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # state -> (session_id, created_monotonic)
        self._pending_oauth: dict[str, tuple[str, float]] = {}
        # session_id -> tokens
        self._sessions: dict[str, INatTokenRecord] = {}

    def new_session_id(self) -> str:
        return secrets.token_urlsafe(32)

    def new_oauth_state(self) -> str:
        return secrets.token_urlsafe(32)

    def register_oauth_state(self, state: str, session_id: str, ttl_seconds: float = 600.0) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune_oauth_locked(now, ttl_seconds)
            self._pending_oauth[state] = (session_id, now)

    def pop_oauth_session(self, state: str, ttl_seconds: float = 600.0) -> Optional[str]:
        now = time.monotonic()
        with self._lock:
            self._prune_oauth_locked(now, ttl_seconds)
            item = self._pending_oauth.pop(state, None)
            if not item:
                return None
            session_id, created = item
            if now - created > ttl_seconds:
                return None
            return session_id

    def _prune_oauth_locked(self, now: float, ttl_seconds: float) -> None:
        stale = [k for k, (_, t) in self._pending_oauth.items() if now - t > ttl_seconds]
        for k in stale:
            self._pending_oauth.pop(k, None)

    def set_tokens(self, session_id: str, record: INatTokenRecord) -> None:
        with self._lock:
            self._sessions[session_id] = record

    def get_tokens(self, session_id: str) -> Optional[INatTokenRecord]:
        with self._lock:
            return self._sessions.get(session_id)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """Test helper / admin: wipe all OAuth state and tokens."""
        with self._lock:
            self._pending_oauth.clear()
            self._sessions.clear()


# Process-wide singleton (tests can replace module attribute)
store = INatSessionStore()
