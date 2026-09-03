"""Access boundaries for Knowledge Center model pickers and global selection."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from fastapi import HTTPException
import pytest

from deeptutor.api.routers import knowledge as knowledge_router
from deeptutor.multi_user import model_access
from deeptutor.multi_user.context import reset_current_user, set_current_user
from deeptutor.multi_user.models import CurrentUser, UserScope

LOCAL_PROFILE = "llm-profile-local"
CURSOR_PROFILE = "llm-profile-cursor"
GROK_PROFILE = "llm-profile-grok"
CODEX_PROFILE = "llm-profile-openai-codex-managed"


def _user(tmp_path: Path, user_id: str, *, role: str = "user") -> CurrentUser:
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


def _catalog() -> dict:
    return {
        model_access.DEPLOYMENT_OWNER_ID_FIELD: "u_owner",
        "services": {
            "llm": {
                "active_profile_id": CURSOR_PROFILE,
                "active_model_id": "cursor-grok",
                "profiles": [
                    {
                        "id": LOCAL_PROFILE,
                        "name": "Local Qwen",
                        "binding": "ollama",
                        "models": [{"id": "qwen", "name": "Qwen", "model": "qwen3.5:4b"}],
                    },
                    {
                        "id": CURSOR_PROFILE,
                        "name": "Cursor Ultra",
                        "binding": "cursor_subscription",
                        "models": [
                            {
                                "id": "cursor-grok",
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
                                "id": "grok-high",
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
                                "id": "codex-luna",
                                "name": "GPT-5.6 Luna",
                                "model": "gpt-5.6-luna",
                                "reasoning_effort": "max",
                            }
                        ],
                    },
                ],
            },
            "embedding": {
                "active_profile_id": "embedding-local",
                "active_model_id": "bge",
                "profiles": [
                    {
                        "id": "embedding-local",
                        "name": "Local embeddings",
                        "binding": "ollama",
                        "models": [
                            {"id": "bge", "name": "BGE", "model": "bge-m3", "dimension": 1024}
                        ],
                    }
                ],
            },
        },
    }


class _CatalogService:
    def __init__(self, catalog: dict) -> None:
        self.catalog = deepcopy(catalog)
        self.applied: list[dict] = []

    def load(self) -> dict:
        return deepcopy(self.catalog)

    def apply(self, catalog: dict) -> None:
        self.catalog = deepcopy(catalog)
        self.applied.append(deepcopy(catalog))


def _install_catalog(monkeypatch: pytest.MonkeyPatch, service: _CatalogService) -> None:
    monkeypatch.setattr(
        "deeptutor.services.config.get_model_catalog_service",
        lambda: service,
    )
    monkeypatch.setattr(model_access, "admin_catalog", service.load)
    monkeypatch.setattr(
        model_access,
        "load_users",
        lambda: {
            "owner@example.com": {"id": "u_owner", "role": "admin"},
            "second@example.com": {"id": "u_second", "role": "admin"},
        },
    )


def test_model_options_show_ordinary_user_exact_granted_subscription_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = _CatalogService(_catalog())
    _install_catalog(monkeypatch, service)
    monkeypatch.setattr(
        model_access,
        "load_grant",
        lambda _user_id=None: {
            "models": {
                "llm": [
                    {
                        "profile_id": CURSOR_PROFILE,
                        "model_ids": ["cursor-grok"],
                        model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD: "u_owner",
                    },
                    {
                        "profile_id": GROK_PROFILE,
                        "model_ids": ["grok-high"],
                        model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD: "u_owner",
                    },
                    {
                        "profile_id": CODEX_PROFILE,
                        "model_ids": ["codex-luna"],
                        model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD: "u_owner",
                    },
                ]
            }
        },
    )
    monkeypatch.setattr(
        "deeptutor.multi_user.personal_models.personal_llm_rows",
        lambda: [],
    )
    token = set_current_user(_user(tmp_path, "u_child"))
    try:
        payload = knowledge_router._model_options_payload(["llm"])
    finally:
        reset_current_user(token)

    assert [(option["profile_id"], option["model_id"]) for option in payload["llm"]["options"]] == [
        (CURSOR_PROFILE, "cursor-grok"),
        (GROK_PROFILE, "grok-high"),
        (CODEX_PROFILE, "codex-luna"),
    ]
    assert payload["llm"]["active"] is None
    assert LOCAL_PROFILE not in str(payload)


def test_model_options_hide_owner_subscription_from_later_admin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = _CatalogService(_catalog())
    _install_catalog(monkeypatch, service)
    token = set_current_user(_user(tmp_path, "u_second", role="admin"))
    try:
        payload = knowledge_router._model_options_payload(["llm"])
    finally:
        reset_current_user(token)

    assert [option["profile_id"] for option in payload["llm"]["options"]] == [LOCAL_PROFILE]
    assert payload["llm"]["active"] is None
    assert CURSOR_PROFILE not in str(payload)
    assert GROK_PROFILE not in str(payload)
    assert CODEX_PROFILE not in str(payload)


@pytest.mark.asyncio
async def test_ordinary_user_cannot_change_global_model_even_when_subscription_is_granted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = _CatalogService(_catalog())
    _install_catalog(monkeypatch, service)
    monkeypatch.setattr(
        model_access,
        "load_grant",
        lambda _user_id=None: {
            "models": {
                "llm": [
                    {
                        "profile_id": CURSOR_PROFILE,
                        "model_ids": ["cursor-grok"],
                        model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD: "u_owner",
                    }
                ]
            }
        },
    )
    token = set_current_user(_user(tmp_path, "u_child"))
    try:
        with pytest.raises(HTTPException) as exc_info:
            await knowledge_router.set_rag_active_model(
                knowledge_router.ActiveModelUpdate(
                    kind="llm", profile_id=CURSOR_PROFILE, model_id="cursor-grok"
                )
            )
    finally:
        reset_current_user(token)

    assert exc_info.value.status_code == 403
    assert service.applied == []


@pytest.mark.asyncio
async def test_later_admin_cannot_change_global_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _CatalogService(_catalog())
    _install_catalog(monkeypatch, service)
    token = set_current_user(_user(tmp_path, "u_second", role="admin"))
    try:
        with pytest.raises(HTTPException) as exc_info:
            await knowledge_router.set_rag_active_model(
                knowledge_router.ActiveModelUpdate(
                    kind="llm", profile_id=CURSOR_PROFILE, model_id="cursor-grok"
                )
            )
    finally:
        reset_current_user(token)

    assert exc_info.value.status_code == 403
    assert service.applied == []


@pytest.mark.asyncio
async def test_deployment_owner_can_change_global_llm_to_subscription_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = _CatalogService(_catalog())
    _install_catalog(monkeypatch, service)
    token = set_current_user(_user(tmp_path, "u_owner", role="admin"))
    try:
        result = await knowledge_router.set_rag_active_model(
            knowledge_router.ActiveModelUpdate(
                kind="llm", profile_id=CURSOR_PROFILE, model_id="cursor-grok"
            )
        )
    finally:
        reset_current_user(token)

    assert result["active"] == {
        "profile_id": CURSOR_PROFILE,
        "model_id": "cursor-grok",
    }
    assert len(service.applied) == 1
