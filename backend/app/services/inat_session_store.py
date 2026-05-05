"""In-memory iNaturalist OAuth state and per-session tokens (swap for Redis/DB later)."""

from __future__ import annotations

import secrets
import sqlite3
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
    """Thread-safe and multi-process store for OAuth CSRF state and session-bound iNat tokens."""

    def __init__(self, db_path: str = "/tmp/lizard_lens_sessions.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pending_oauth (
                    state TEXT PRIMARY KEY,
                    session_id TEXT,
                    created_at REAL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    access_token TEXT,
                    refresh_token TEXT,
                    expires_at REAL
                )
            ''')
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def new_session_id(self) -> str:
        return secrets.token_urlsafe(32)

    def new_oauth_state(self) -> str:
        return secrets.token_urlsafe(32)

    def register_oauth_state(self, state: str, session_id: str, ttl_seconds: float = 600.0) -> None:
        now = time.time()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute('DELETE FROM pending_oauth WHERE ? - created_at > ?', (now, ttl_seconds))
                conn.execute('INSERT OR REPLACE INTO pending_oauth (state, session_id, created_at) VALUES (?, ?, ?)',
                             (state, session_id, now))
                conn.commit()

    def pop_oauth_session(self, state: str, ttl_seconds: float = 600.0) -> Optional[str]:
        now = time.time()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute('DELETE FROM pending_oauth WHERE ? - created_at > ?', (now, ttl_seconds))
                cur = conn.execute('SELECT session_id, created_at FROM pending_oauth WHERE state = ?', (state,))
                row = cur.fetchone()
                if row:
                    conn.execute('DELETE FROM pending_oauth WHERE state = ?', (state,))
                    conn.commit()
                    session_id, created_at = row
                    if now - created_at <= ttl_seconds:
                        return session_id
                conn.commit()
        return None

    def set_tokens(self, session_id: str, record: INatTokenRecord) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO sessions (session_id, access_token, refresh_token, expires_at) VALUES (?, ?, ?, ?)',
                    (session_id, record.access_token, record.refresh_token, record.expires_at))
                conn.commit()

    def get_tokens(self, session_id: str) -> Optional[INatTokenRecord]:
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute('SELECT access_token, refresh_token, expires_at FROM sessions WHERE session_id = ?', (session_id,))
                row = cur.fetchone()
                if row:
                    return INatTokenRecord(access_token=row[0], refresh_token=row[1], expires_at=row[2])
        return None

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
                conn.commit()

    def clear_all(self) -> None:
        """Test helper / admin: wipe all OAuth state and tokens."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute('DELETE FROM pending_oauth')
                conn.execute('DELETE FROM sessions')
                conn.commit()


# Process-wide singleton (tests can replace module attribute)
store = INatSessionStore()
