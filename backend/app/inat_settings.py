"""iNaturalist OAuth and cookie settings (env: INAT_*)."""

from functools import lru_cache

# HTTP-only session cookie for server-stored iNaturalist tokens
INAT_SESSION_COOKIE_NAME = "anole_inat_session"

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class INatOAuthSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INAT_",
        env_file=".env",
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


class AppAuthFlags(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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


def _is_https(url: str) -> bool:
    return url.lower().startswith("https://")


def session_cookie_secure() -> bool:
    """Use Secure cookies when using HTTPS callback or SameSite=None."""
    s = get_inat_oauth_settings()
    if s.cookie_samesite.lower() == "none":
        return True
    return _is_https(s.redirect_uri) or _is_https(s.frontend_success_url)
