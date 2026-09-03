"""Cloudflare Access header → DeepTutor user SSO."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest


@pytest.fixture()
def auth_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPTUTOR_HOME", str(tmp_path))
    # Import after HOME is set so auth secret / users land under tmp_path.
    from deeptutor.api.routers import auth as auth_router
    from deeptutor.services import auth as auth_service
    from deeptutor.services.config import ensure_runtime_settings_files

    ensure_runtime_settings_files()
    monkeypatch.setattr(auth_service, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth_service, "POCKETBASE_ENABLED", False)
    monkeypatch.setattr(auth_router, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth_router, "POCKETBASE_ENABLED", False)
    monkeypatch.setattr(auth_router, "_SECURE", False)
    monkeypatch.setattr(auth_router, "_SAMESITE", "lax")

    # Fresh secret for JWT minting in this temp home.
    from deeptutor.multi_user.identity import load_or_create_auth_secret

    secret = load_or_create_auth_secret()
    monkeypatch.setattr(auth_service, "AUTH_SECRET", secret)

    auth_service.add_user("Fol2hk@gmail.com", "AdminPass123!", role="admin")
    auth_service.add_user("wingkwokshan@gmail.com", "UserPass123!", role="user")
    auth_service.add_user("eugeniayyto@gmail.com", "UserPass123!", role="user")
    auth_service.add_user("nelsonnsto@gmail.com", "UserPass123!", role="user")

    from deeptutor.api.main import app

    return TestClient(app)


def test_find_user_by_email_is_case_insensitive(auth_app, monkeypatch):
    from deeptutor.services import auth as auth_service

    monkeypatch.setattr(auth_service, "AUTH_ENABLED", True)
    payload = auth_service.find_user_by_email("fol2hk@gmail.com")
    assert payload is not None
    assert payload.username == "Fol2hk@gmail.com"
    assert payload.role == "admin"


@pytest.mark.parametrize(
    "email,role",
    [
        ("Fol2hk@gmail.com", "admin"),
        ("wingkwokshan@gmail.com", "user"),
        ("eugeniayyto@gmail.com", "user"),
        ("nelsonnsto@gmail.com", "user"),
    ],
)
def test_access_sso_sets_cookie_for_provisioned_users(auth_app, email, role):
    response = auth_app.get(
        "/api/v1/auth/sso/access",
        params={"next": "/home"},
        headers={"Cf-Access-Authenticated-User-Email": email},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/home"
    assert "dt_token=" in response.headers.get("set-cookie", "")


def test_access_sso_rejects_unknown_email(auth_app):
    response = auth_app.get(
        "/api/v1/auth/sso/access",
        headers={"Cf-Access-Authenticated-User-Email": "stranger@example.com"},
        follow_redirects=False,
    )
    assert response.status_code == 403


def test_require_auth_accepts_access_header(auth_app):
    response = auth_app.get(
        "/api/v1/auth/status",
        headers={"Cf-Access-Authenticated-User-Email": "eugeniayyto@gmail.com"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["authenticated"] is True
    assert body["username"] == "eugeniayyto@gmail.com"
    assert body["role"] == "user"
    assert body["is_admin"] is False


def test_access_sso_trusted_by_jwt_assertion_when_peer_is_public():
    from deeptutor.api.routers.auth import _access_sso_trusted

    request = MagicMock()
    request.client = MagicMock(host="2a04:204:4e62:5100::1")
    request.headers = {
        "Cf-Access-Jwt-Assertion": "eyJhbGciOiJSUzI1NiJ9.fake",
    }
    assert _access_sso_trusted(request) is True


def test_access_sso_rejects_public_peer_without_assertion():
    from deeptutor.api.routers.auth import _access_sso_trusted

    request = MagicMock()
    request.client = MagicMock(host="2a04:204:4e62:5100::1")
    request.headers = {}
    assert _access_sso_trusted(request) is False
