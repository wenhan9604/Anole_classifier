"""iNaturalist OAuth: state validation and callback error handling."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def configured_client(monkeypatch, clear_inat_settings_cache):
    monkeypatch.setenv("INAT_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("INAT_CLIENT_SECRET", "test_secret")
    monkeypatch.setenv("INAT_REDIRECT_URI", "https://api.example.com/api/auth/inat/callback")
    monkeypatch.setenv("INAT_FRONTEND_SUCCESS_URL", "https://app.example.com/predict?inat=connected")
    monkeypatch.setenv("INAT_COOKIE_SAMESITE", "none")
    from app.main import app

    return TestClient(app)


def test_inat_login_returns_503_when_not_configured(monkeypatch, clear_inat_settings_cache):
    monkeypatch.delenv("INAT_CLIENT_ID", raising=False)
    monkeypatch.delenv("INAT_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("INAT_REDIRECT_URI", raising=False)
    monkeypatch.delenv("INAT_FRONTEND_SUCCESS_URL", raising=False)
    from app.main import app

    r = TestClient(app).get("/api/auth/inat/login", follow_redirects=False)
    assert r.status_code == 503


def test_inat_callback_missing_params(configured_client: TestClient):
    r = configured_client.get("/api/auth/inat/callback", follow_redirects=False)
    assert r.status_code == 400


def test_inat_callback_invalid_state(configured_client: TestClient):
    r = configured_client.get(
        "/api/auth/inat/callback",
        params={"code": "abc", "state": "not_registered"},
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "state" in r.json().get("detail", "").lower() or "invalid" in r.json().get("detail", "").lower()


def test_inat_callback_token_exchange_http_error(configured_client: TestClient):
    from app.services.inat_session_store import store

    state = "test_state_value"
    session_id = "test_session_id"
    store.register_oauth_state(state, session_id)

    with patch(
        "app.routers.auth.inat_oauth_client.exchange_code_for_tokens",
        new_callable=AsyncMock,
        side_effect=__import__("httpx").HTTPStatusError(
            "bad",
            request=__import__("httpx").Request("POST", "https://www.inaturalist.org/oauth/token"),
            response=__import__("httpx").Response(401, text="invalid_grant"),
        ),
    ):
        r = configured_client.get(
            "/api/auth/inat/callback",
            params={"code": "exchange_will_fail", "state": state},
            follow_redirects=False,
        )
    assert r.status_code == 400


def test_inat_status_not_connected(configured_client: TestClient):
    r = configured_client.get("/api/auth/inat/status")
    assert r.status_code == 200
    assert r.json() == {"connected": False, "expiresAt": None}


def test_mock_login_disabled_by_default(monkeypatch, clear_inat_settings_cache):
    monkeypatch.delenv("ENABLE_INAT_MOCK_AUTH", raising=False)
    from app.main import app

    r = TestClient(app).post("/api/auth/mock-login")
    assert r.status_code == 404


def test_mock_login_when_enabled(monkeypatch, clear_inat_settings_cache):
    monkeypatch.setenv("ENABLE_INAT_MOCK_AUTH", "true")
    from app.main import app

    r = TestClient(app).post("/api/auth/mock-login")
    assert r.status_code == 200
    body = r.json()
    assert "accessToken" in body


def test_observations_requires_inat_session(configured_client: TestClient):
    """POST /api/observations without OAuth cookie returns 401."""
    files = {"file": ("x.jpg", b"\xff\xd8\xff", "image/jpeg")}
    data = {
        "species": "Green Anole",
        "scientificName": "Anolis carolinensis",
        "confidence": "0.9",
        "count": "1",
    }
    r = configured_client.post("/api/observations", files=files, data=data)
    assert r.status_code == 401
