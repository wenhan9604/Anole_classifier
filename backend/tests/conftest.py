import os

import pytest


def pytest_configure(config):
    """Avoid loading backend/.env during tests (would override monkeypatched env)."""
    os.environ["ANOLE_SKIP_DOTENV"] = "1"


@pytest.fixture(autouse=True)
def reset_inat_session_store():
    from app.services.inat_session_store import store

    store.clear_all()
    yield
    store.clear_all()


@pytest.fixture(autouse=True)
def clear_inat_settings_cache_before_each_test():
    """Settings are lru_cached; clear so each test's monkeypatched env is applied."""
    from app.inat_settings import get_app_auth_flags, get_inat_oauth_settings

    get_inat_oauth_settings.cache_clear()
    get_app_auth_flags.cache_clear()
    yield


@pytest.fixture
def clear_inat_settings_cache():
    """Explicit clear for tests that re-import app after env changes."""
    from app.inat_settings import get_app_auth_flags, get_inat_oauth_settings

    get_inat_oauth_settings.cache_clear()
    get_app_auth_flags.cache_clear()
    yield
    get_inat_oauth_settings.cache_clear()
    get_app_auth_flags.cache_clear()
