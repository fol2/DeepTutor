"""Only the deployment owner may drive the Codex OAuth lifecycle.

Codex authenticates one adult operator's consumer subscription. Its credential,
managed profile, model refresh, and reasoning overrides are deployment-owner
state: child accounts, ordinary users, partners, and later administrators must
never reach the lifecycle service.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from deeptutor.api.routers import settings as settings_router
from deeptutor.multi_user import model_access
from deeptutor.multi_user.models import CurrentUser, UserScope
from deeptutor.services.codex_auth.contracts import CatalogSnapshot, CodexModel
from deeptutor.services.codex_auth.service import (
    CodexOAuthService,
    remove_codex_catalog,
    sync_codex_catalog,
)
from deeptutor.services.codex_auth.storage import CodexCredentialStore
from deeptutor.services.config.model_catalog import ModelCatalogService
from deeptutor.services.partners.scope import PARTNER_USER_PREFIX

CODEX_ROUTES = [
    ("post", "/api/v1/settings/providers/openai-codex/oauth/start"),
    ("get", "/api/v1/settings/providers/openai-codex/oauth/status"),
    ("post", "/api/v1/settings/providers/openai-codex/oauth/cancel"),
    ("post", "/api/v1/settings/providers/openai-codex/oauth/logout"),
    ("post", "/api/v1/settings/providers/openai-codex/models/refresh"),
]


class _Service:
    """Stand-in for the deployment owner's ``CodexOAuthService``."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def start_login(self) -> dict[str, Any]:
        self.calls.append("start")
        return {"operation_id": "op-1"}

    def public_status(self) -> dict[str, Any]:
        self.calls.append("status")
        return {"connection": "disconnected"}

    async def cancel_login(self) -> dict[str, Any]:
        self.calls.append("cancel")
        return {"connection": "disconnected"}

    async def logout(self) -> dict[str, Any]:
        self.calls.append("logout")
        return {"connection": "disconnected"}

    async def refresh_models(self) -> dict[str, Any]:
        self.calls.append("refresh")
        return {"connection": "connected"}

    async def set_reasoning_effort(
        self,
        model: str,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        self.calls.append(f"reasoning:{model}:{reasoning_effort}")
        return {"connection": "connected"}


def _user(uid: str, *, role: str, root) -> CurrentUser:
    return CurrentUser(
        id=uid,
        username=uid,
        role=role,
        scope=UserScope(kind="user", user_id=uid, root=root),
    )


@pytest.fixture
def client(tmp_path, monkeypatch) -> tuple[TestClient, _Service, dict[str, CurrentUser]]:
    service = _Service()
    owner = _user("u_owner", role="admin", root=tmp_path / "owner")
    current: dict[str, CurrentUser] = {"user": owner}
    owner_catalog = {
        "deployment_owner_user_id": owner.id,
        "services": {"llm": {"profiles": []}},
    }

    monkeypatch.setattr(settings_router, "get_codex_oauth_service", lambda: service)
    monkeypatch.setattr(settings_router, "get_current_user", lambda: current["user"])
    monkeypatch.setattr(model_access, "get_current_user", lambda: current["user"])
    monkeypatch.setattr(model_access, "admin_catalog", lambda: owner_catalog)
    monkeypatch.setattr(
        model_access,
        "load_users",
        lambda: {
            "owner@example.test": {
                "id": owner.id,
                "role": "admin",
                "disabled": False,
            }
        },
    )

    app = FastAPI()
    app.include_router(settings_router.router, prefix="/api/v1/settings")
    return TestClient(app), service, current


@pytest.mark.parametrize(("method", "path"), CODEX_ROUTES)
def test_deployment_owner_can_drive_codex_lifecycle(client, method, path) -> None:
    test_client, service, _current = client

    response = getattr(test_client, method)(path)

    assert response.status_code == 200
    assert service.calls, "the request must reach the deployment-owner service"


def test_deployment_owner_can_set_codex_reasoning_effort(client) -> None:
    test_client, service, _current = client

    response = test_client.post(
        "/api/v1/settings/providers/openai-codex/models/reasoning-effort",
        json={"model": "gpt-5.6-sol", "reasoning_effort": "high"},
    )

    assert response.status_code == 200
    assert service.calls == ["reasoning:gpt-5.6-sol:high"]


@pytest.mark.parametrize(("method", "path"), CODEX_ROUTES)
@pytest.mark.parametrize(
    ("uid", "role"),
    [
        ("u_child", "user"),
        ("u_later_admin", "admin"),
        (f"{PARTNER_USER_PREFIX}ada", "user"),
    ],
)
def test_non_owner_cannot_drive_codex_lifecycle(client, tmp_path, method, path, uid, role) -> None:
    test_client, service, current = client
    current["user"] = _user(uid, role=role, root=tmp_path / uid)

    response = getattr(test_client, method)(path)

    assert response.status_code == 403
    assert "deployment owner" in response.json()["detail"].lower()
    assert service.calls == []


@pytest.mark.parametrize(
    ("uid", "role"),
    [
        ("u_child", "user"),
        ("u_later_admin", "admin"),
        (f"{PARTNER_USER_PREFIX}ada", "user"),
    ],
)
def test_non_owner_cannot_change_codex_reasoning_effort(client, tmp_path, uid, role) -> None:
    test_client, service, current = client
    current["user"] = _user(uid, role=role, root=tmp_path / uid)

    response = test_client.post(
        "/api/v1/settings/providers/openai-codex/models/reasoning-effort",
        json={"model": "gpt-5.6-sol", "reasoning_effort": "high"},
    )

    assert response.status_code == 403
    assert "deployment owner" in response.json()["detail"].lower()
    assert service.calls == []


def test_pocketbase_first_codex_login_preserves_owner_lifecycle_access(
    tmp_path, monkeypatch
) -> None:
    """Publishing the first managed profile must claim the catalogue atomically."""
    from deeptutor.services import auth as auth_service

    owner = _user("pb_owner", role="admin", root=tmp_path / "owner")
    current: dict[str, CurrentUser] = {"user": owner}
    catalog_service = ModelCatalogService(tmp_path / "model_catalog.json")
    snapshot = CatalogSnapshot(
        models=(
            CodexModel(
                slug="gpt-5.6-sol",
                display_name="GPT-5.6 Sol",
                priority=1,
                visibility="list",
                default_reasoning_level="high",
                supported_reasoning_levels=("high",),
                supports_reasoning_summary=True,
                supports_parallel_tool_calls=True,
                use_responses_lite=False,
            ),
        ),
        source="live",
        fetched_at=1_000,
        etag=None,
        generation=1,
        account_hash="account-hash",
    )

    class _FirstLoginService(_Service):
        async def start_login(self) -> dict[str, Any]:
            self.calls.append("start")
            sync_codex_catalog(
                catalog_service,
                snapshot,
                account_id="account-1",
                deployment_owner_user_id=current["user"].id,
            )
            return {"operation_id": "op-1"}

        async def refresh_models(self) -> dict[str, Any]:
            self.calls.append("refresh")
            sync_codex_catalog(
                catalog_service,
                snapshot,
                account_id="account-1",
                deployment_owner_user_id=current["user"].id,
            )
            return {"connection": "connected"}

        async def logout(self) -> dict[str, Any]:
            self.calls.append("logout")
            remove_codex_catalog(catalog_service)
            return {"connection": "disconnected"}

    service = _FirstLoginService()
    monkeypatch.setattr(auth_service, "POCKETBASE_ENABLED", True)
    monkeypatch.setattr(model_access, "load_users", lambda: {})
    monkeypatch.setattr(model_access, "get_current_user", lambda: current["user"])
    monkeypatch.setattr("deeptutor.multi_user.context.get_current_user", lambda: current["user"])
    monkeypatch.setattr(model_access, "admin_catalog", catalog_service.load)
    monkeypatch.setattr(settings_router, "get_current_user", lambda: current["user"])
    monkeypatch.setattr(settings_router, "get_model_catalog_service", lambda: catalog_service)
    monkeypatch.setattr(settings_router, "get_codex_oauth_service", lambda: service)

    app = FastAPI()
    app.include_router(settings_router.router, prefix="/api/v1/settings")
    test_client = TestClient(app)

    assert test_client.post(CODEX_ROUTES[0][1]).status_code == 200
    assert catalog_service.load()[model_access.DEPLOYMENT_OWNER_ID_FIELD] == owner.id
    current["user"] = _user("pb_second", role="admin", root=tmp_path / "second")
    for method, path in CODEX_ROUTES:
        assert getattr(test_client, method)(path).status_code == 403
    assert service.calls == ["start"]
    current["user"] = owner
    assert test_client.get(CODEX_ROUTES[1][1]).status_code == 200
    assert test_client.post(CODEX_ROUTES[4][1]).status_code == 200
    settings_response = test_client.get("/api/v1/settings")
    assert settings_response.status_code == 200
    assert settings_response.json()["catalog"][model_access.DEPLOYMENT_OWNER_ID_FIELD] == owner.id
    assert test_client.post(CODEX_ROUTES[3][1]).status_code == 200
    assert test_client.get("/api/v1/settings").status_code == 200
    assert catalog_service.load()[model_access.DEPLOYMENT_OWNER_ID_FIELD] == owner.id
    assert service.calls == ["start", "status", "refresh", "logout"]


def test_second_pocketbase_admin_cannot_join_pending_codex_login(tmp_path, monkeypatch) -> None:
    """The first start claims ownership before exposing the shared operation."""
    from deeptutor.services import auth as auth_service

    class _PendingCallback:
        port = 1455

        async def wait(self, *, timeout: float):
            del timeout
            await asyncio.Event().wait()

        async def cancel(self) -> None:
            return None

    async def callback_factory(_state: str) -> _PendingCallback:
        return _PendingCallback()

    owner = _user("pb_owner", role="admin", root=tmp_path / "owner")
    second = _user("pb_second", role="admin", root=tmp_path / "second")
    current: dict[str, CurrentUser] = {"user": owner}
    catalog_service = ModelCatalogService(tmp_path / "pending-model-catalog.json")
    service = CodexOAuthService(
        CodexCredentialStore(tmp_path / "secrets"),
        object(),  # type: ignore[arg-type]
        catalog_service,
        oauth_client=object(),  # type: ignore[arg-type]
        callback_factory=callback_factory,
    )
    monkeypatch.setattr(auth_service, "POCKETBASE_ENABLED", True)
    monkeypatch.setattr(model_access, "load_users", lambda: {})
    monkeypatch.setattr(model_access, "get_current_user", lambda: current["user"])
    monkeypatch.setattr("deeptutor.multi_user.context.get_current_user", lambda: current["user"])
    monkeypatch.setattr(model_access, "admin_catalog", catalog_service.load)
    monkeypatch.setattr(settings_router, "get_current_user", lambda: current["user"])
    monkeypatch.setattr(settings_router, "get_model_catalog_service", lambda: catalog_service)
    monkeypatch.setattr(settings_router, "get_codex_oauth_service", lambda: service)

    app = FastAPI()
    app.include_router(settings_router.router, prefix="/api/v1/settings")
    with TestClient(app) as test_client:
        first_response = test_client.post(CODEX_ROUTES[0][1])
        assert first_response.status_code == 200
        assert first_response.json()["operation_id"]
        assert catalog_service.load()[model_access.DEPLOYMENT_OWNER_ID_FIELD] == owner.id

        current["user"] = second
        blocked = [getattr(test_client, method)(path) for method, path in CODEX_ROUTES]
        assert all(response.status_code == 403 for response in blocked)
        blocked_payloads = json.dumps([response.json() for response in blocked])
        assert "authorize_url" not in blocked_payloads
        assert "operation_id" not in blocked_payloads
        second_settings = test_client.get("/api/v1/settings")
        assert second_settings.status_code == 200
        assert model_access.DEPLOYMENT_OWNER_ID_FIELD not in second_settings.json()["catalog"]

        current["user"] = owner
        assert test_client.post(CODEX_ROUTES[2][1]).status_code == 200
        assert catalog_service.load()[model_access.DEPLOYMENT_OWNER_ID_FIELD] == owner.id
