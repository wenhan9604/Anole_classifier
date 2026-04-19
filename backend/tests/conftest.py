import pytest


@pytest.fixture(autouse=True)
def reset_inat_session_store():
    from app.services.inat_session_store import store

    store.clear_all()
    yield
    store.clear_all()


@pytest.fixture
def clear_inat_settings_cache():
    from app.inat_settings import get_app_auth_flags, get_inat_oauth_settings

    get_inat_oauth_settings.cache_clear()
    get_app_auth_flags.cache_clear()
    yield
    get_inat_oauth_settings.cache_clear()
    get_app_auth_flags.cache_clear()
