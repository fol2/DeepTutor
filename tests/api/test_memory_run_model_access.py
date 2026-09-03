"""Memory runs may use only the request owner's authorised LLM models."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from deeptutor.multi_user import model_access, personal_models
from deeptutor.multi_user.context import reset_current_user, set_current_user
from deeptutor.multi_user.models import CurrentUser, UserScope

memory_router = importlib.import_module("deeptutor.api.routers.memory")
runs_module = importlib.import_module("deeptutor.services.memory.consolidator.runs")

LOCAL_PROFILE = "llm-profile-local"
LOCAL_GRANTED_MODEL = "llm-model-qwen-granted"
LOCAL_UNGRANTED_MODEL = "llm-model-qwen-ungranted"
CURSOR_PROFILE = "llm-profile-cursor-subscription"
CURSOR_MODEL = "llm-model-cursor-grok-high"
CURSOR_UNGRANTED_MODEL = "llm-model-cursor-not-granted"
GROK_PROFILE = "llm-profile-grok-subscription"
GROK_MODEL = "llm-model-grok-high"
CODEX_PROFILE = "llm-profile-openai-codex-managed"
CODEX_MODEL = "llm-model-openai-codex-luna"


def _catalog(*, owner_id: str = "u_owner") -> dict[str, Any]:
    return {
        "deployment_owner_user_id": owner_id,
        "services": {
            "llm": {
                "active_profile_id": CURSOR_PROFILE,
                "active_model_id": CURSOR_MODEL,
                "profiles": [
                    {
                        "id": LOCAL_PROFILE,
                        "name": "Local Qwen",
                        "binding": "ollama",
                        "models": [
                            {
                                "id": LOCAL_GRANTED_MODEL,
                                "name": "Qwen granted",
                                "model": "qwen3.5:4b",
                            },
                            {
                                "id": LOCAL_UNGRANTED_MODEL,
                                "name": "Qwen not granted",
                                "model": "qwen3.5:9b",
                            },
                        ],
                    },
                    {
                        "id": CURSOR_PROFILE,
                        "name": "Cursor Ultra",
                        "binding": "cursor_subscription",
                        "models": [
                            {
                                "id": CURSOR_MODEL,
                                "name": "Grok 4.6 High",
                                "model": "cursor-grok-4.6-high",
                            }
                        ],
                    },
                    {
                        "id": GROK_PROFILE,
                        "name": "SuperGrok Heavy",
                        "binding": "grok_subscription",
                        "models": [
                            {
                                "id": GROK_MODEL,
                                "name": "Grok 4.6 High",
                                "model": "grok-4.6-high",
                            }
                        ],
                    },
                    {
                        "id": CODEX_PROFILE,
                        "name": "ChatGPT Pro",
                        "binding": "openai_codex",
                        "managed_by": "openai_codex_oauth",
                        "models": [
                            {
                                "id": CODEX_MODEL,
                                "name": "GPT-5.6 Luna",
                                "model": "gpt-5.6-luna",
                                "reasoning_effort": "max",
                            }
                        ],
                    },
                ],
            }
        },
    }


def _family_grant(_user_id: str | None = None) -> dict[str, Any]:
    """Grant the exact three owner-managed family subscription models."""
    return {
        "models": {
            "llm": [
                {
                    "profile_id": CURSOR_PROFILE,
                    "model_ids": [CURSOR_MODEL],
                    model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD: "u_owner",
                },
                {
                    "profile_id": GROK_PROFILE,
                    "model_ids": [GROK_MODEL],
                    model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD: "u_owner",
                },
                {
                    "profile_id": CODEX_PROFILE,
                    "model_ids": [CODEX_MODEL],
                    model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD: "u_owner",
                },
            ]
        }
    }


def _user(tmp_path: Path, *, user_id: str = "u_child", role: str = "user") -> CurrentUser:
    return CurrentUser(
        id=user_id,
        username=f"{user_id}@example.com",
        role=role,  # type: ignore[arg-type]
        scope=UserScope(
            kind="admin" if role == "admin" else "user",
            user_id=user_id,
            root=tmp_path / user_id,
        ),
    )


@dataclass
class _FakeRun:
    call: dict[str, Any]
    active: bool = False

    @property
    def events(self) -> list[Any]:
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": "run-test",
            "layer": self.call["layer"],
            "key": self.call["key"],
            "mode": self.call["mode"],
            "params": self.call["params"],
            "status": "queued",
        }


class _CapturingRunManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start(self, **kwargs: Any) -> _FakeRun:
        self.calls.append(kwargs)
        return _FakeRun(kwargs)

    async def wait_for_events(self, _run: _FakeRun, *, since: int) -> list[Any]:
        assert since == 0
        return []


@pytest.fixture
def memory_api(monkeypatch: pytest.MonkeyPatch):
    manager = _CapturingRunManager()
    current: dict[str, CurrentUser | None] = {"user": None}

    async def never_run(_on_event):
        raise AssertionError("a model-access test must not launch consolidator work")

    monkeypatch.setattr(runs_module, "get_run_manager", lambda: manager)
    monkeypatch.setattr(memory_router, "_runner_for", lambda _req: never_run)
    monkeypatch.setattr(model_access, "admin_catalog", _catalog)
    monkeypatch.setattr(model_access, "load_grant", _family_grant)
    monkeypatch.setattr(
        model_access,
        "load_users",
        lambda: {
            "owner@example.com": {
                "id": "u_owner",
                "role": "admin",
                "disabled": False,
            }
        },
    )
    monkeypatch.setattr(personal_models, "personal_llm_rows", lambda: [])

    app = FastAPI()

    @app.middleware("http")
    async def bind_current_user(request, call_next):
        assert current["user"] is not None
        token = set_current_user(current["user"])
        try:
            return await call_next(request)
        finally:
            reset_current_user(token)

    app.include_router(memory_router.router, prefix="/api/v1/memory")
    with TestClient(app) as client:
        yield client, manager, current


@pytest.mark.parametrize(
    ("profile_id", "model_id"),
    [
        (CURSOR_PROFILE, CURSOR_MODEL),
        (GROK_PROFILE, GROK_MODEL),
        (CODEX_PROFILE, CODEX_MODEL),
    ],
)
def test_ordinary_user_can_start_run_with_exact_granted_subscription(
    memory_api,
    tmp_path: Path,
    profile_id: str,
    model_id: str,
) -> None:
    client, manager, current = memory_api
    current["user"] = _user(tmp_path)

    response = client.post(
        "/api/v1/memory/runs/start",
        json={
            "layer": "L2",
            "key": "chat",
            "mode": "update",
            "llm_selection": {"profile_id": profile_id, "model_id": model_id},
        },
    )

    assert response.status_code == 200
    expected = {"profile_id": profile_id, "model_id": model_id}
    assert response.json()["params"]["llm_selection"] == expected
    assert len(manager.calls) == 1
    assert manager.calls[0]["params"]["llm_selection"] == expected


@pytest.mark.parametrize(
    ("profile_id", "model_id"),
    [
        (LOCAL_PROFILE, LOCAL_GRANTED_MODEL),
        (CURSOR_PROFILE, CURSOR_UNGRANTED_MODEL),
    ],
)
def test_ordinary_user_cannot_start_run_with_ungranted_or_wrong_model(
    memory_api,
    tmp_path: Path,
    profile_id: str,
    model_id: str,
) -> None:
    client, manager, current = memory_api
    current["user"] = _user(tmp_path)

    response = client.post(
        "/api/v1/memory/runs/start",
        json={
            "layer": "L2",
            "key": "chat",
            "mode": "audit",
            "llm_selection": {
                "profile_id": profile_id,
                "model_id": model_id,
            },
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "This model is not assigned to your account."
    assert manager.calls == []


def test_no_selection_is_pinned_to_first_granted_model_before_run_creation(
    memory_api, tmp_path: Path
) -> None:
    client, manager, current = memory_api
    current["user"] = _user(tmp_path)

    response = client.post(
        "/api/v1/memory/runs/start",
        json={"layer": "L2", "key": "chat", "mode": "dedup"},
    )

    assert response.status_code == 200
    expected = {"profile_id": CURSOR_PROFILE, "model_id": CURSOR_MODEL}
    assert response.json()["params"]["llm_selection"] == expected
    assert len(manager.calls) == 1
    assert manager.calls[0]["params"]["llm_selection"] == expected


def test_legacy_wrapper_pins_no_selection_before_run_creation(memory_api, tmp_path: Path) -> None:
    client, manager, current = memory_api
    current["user"] = _user(tmp_path)

    response = client.post("/api/v1/memory/doc/L2/chat/update", json={"language": "en"})

    assert response.status_code == 200
    assert len(manager.calls) == 1
    assert manager.calls[0]["params"]["llm_selection"] == {
        "profile_id": CURSOR_PROFILE,
        "model_id": CURSOR_MODEL,
    }


def test_deployment_owner_can_start_run_with_active_subscription(
    memory_api, tmp_path: Path
) -> None:
    client, manager, current = memory_api
    current["user"] = _user(tmp_path, user_id="u_owner", role="admin")

    response = client.post(
        "/api/v1/memory/runs/start",
        json={"layer": "L2", "key": "chat", "mode": "update"},
    )

    assert response.status_code == 200
    assert len(manager.calls) == 1
    # No override is necessary: the deployment-owner check validated the
    # catalog's active Cursor profile and the runner will resolve that profile.
    assert manager.calls[0]["params"]["llm_selection"] is None
