"""iNaturalist OAuth and cookie settings (env: INAT_*)."""

from functools import lru_cache

# HTTP-only session cookie for server-stored iNaturalist tokens
INAT_SESSION_COOKIE_NAME = "anole_inat_session"

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class INatOAuthSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INAT_",
        extra="ignore",
    )

    client_id: str = Field(default="", description="iNaturalist OAuth application ID")
    client_secret: str = Field(default="", description="iNaturalist OAuth secret")
    redirect_uri: str = Field(
        default="",
        description="Registered callback URL (e.g. https://api.example.com/api/auth/inat/callback)",
    )
    scopes: str = Field(
        default="",
        description="Optional space-separated OAuth scopes",
    )
    frontend_success_url: str = Field(
        default="",
        description="Browser redirect after successful OAuth (e.g. https://app.example.com/predict?inat=connected)",
    )
    cookie_samesite: str = Field(
        default="lax",
        description="Session cookie SameSite: lax | strict | none",
    )
    cookie_secure: str = Field(
        default="auto",
        description="Session cookie Secure: auto (HTTPS request only), true, or false",
    )


class AppAuthFlags(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enable_inat_mock_auth: bool = Field(
        default=False,
        description="If true, POST /api/auth/mock-login is enabled (dev only). Env: ENABLE_INAT_MOCK_AUTH",
    )


@lru_cache
def get_inat_oauth_settings() -> INatOAuthSettings:
    return INatOAuthSettings()


@lru_cache
def get_app_auth_flags() -> AppAuthFlags:
    return AppAuthFlags()


def inat_oauth_configured() -> bool:
    s = get_inat_oauth_settings()
    return bool(s.client_id and s.client_secret and s.redirect_uri and s.frontend_success_url)


def session_cookie_secure(request_scheme: str, x_forwarded_proto: str | None) -> bool:
    """
    Whether to set the session cookie with the Secure flag.
    Prefer the actual request (or reverse-proxy) scheme over INAT_* URLs so
    local http://localhost:8000 still gets a usable cookie when redirect_uri is https.
    """
    s = get_inat_oauth_settings()
    mode = (s.cookie_secure or "auto").strip().lower()
    if mode == "true":
        return True
    if mode == "false":
        return False
    # auto
    if s.cookie_samesite.lower() == "none":
        return True
    if x_forwarded_proto:
        first = x_forwarded_proto.split(",")[0].strip().lower()
        if first == "https":
            return True
    return (request_scheme or "").lower() == "https"
