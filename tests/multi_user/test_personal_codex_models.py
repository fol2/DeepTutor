"""Legacy per-user Codex profiles never become an ordinary user's models.

Consumer-subscription credentials are deployment-owner state. Older releases
could leave a managed Codex profile in an ordinary user's private catalogue;
that historical file must not grant model access, reach selection validation,
or overlay the deployment catalogue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from deeptutor.multi_user import model_access, personal_models

CODEX_PROFILE = "llm-profile-openai-codex-managed"
SHARED_PROFILE = "llm-profile-shared-key"


def _codex_profile() -> dict[str, Any]:
    return {
        "id": CODEX_PROFILE,
        "name": "OpenAI Codex",
        "binding": "openai_codex",
        "api_key": "",
        "owner_bound": True,
        "read_only": True,
        "managed_by": "openai_codex_oauth",
        "models": [
            {
                "id": "m-sol",
                "name": "GPT 5.6 Sol",
                "model": "gpt-5.6-sol",
            }
        ],
    }


def _write_catalog(path: Path, profiles: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "services": {"llm": {"profiles": profiles}}}),
        encoding="utf-8",
    )


def _write_legacy_personal_codex(as_user, uid: str) -> None:
    with as_user(uid):
        _write_catalog(personal_models.owner_catalog_service().path, [_codex_profile()])


def _admin_catalog(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    return {"services": {"llm": {"profiles": profiles}}}


@pytest.fixture
def no_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_access, "load_grant", lambda _uid=None: {"models": {"llm": []}})


def test_personal_catalog_path_remains_isolated_from_the_shared_catalog(
    as_user,
    mu_isolated_root: Path,
) -> None:
    """The legacy path stays isolated even though its LLM profiles are ignored."""
    with as_user("u_alice"):
        alice_path = personal_models.owner_catalog_service().path
    with as_user("root", role="admin"):
        admin_path = personal_models.owner_catalog_service().path

    assert alice_path != admin_path
    assert "users/u_alice" in alice_path.as_posix()
    assert alice_path.is_relative_to((mu_isolated_root / "data" / "users" / "u_alice").resolve())


def test_legacy_personal_codex_profile_grants_no_ordinary_user_access(
    as_user,
    monkeypatch: pytest.MonkeyPatch,
    no_grants: None,
) -> None:
    monkeypatch.setattr(model_access, "admin_catalog", lambda: _admin_catalog([]))
    _write_legacy_personal_codex(as_user, "u_alice")

    with as_user("u_alice"):
        assert personal_models.personal_llm_rows() == []
        assert model_access.redacted_model_access()["llm"] == []
        assert model_access.allowed_llm_options()["options"] == []
        assert model_access.has_capability_access("llm") is False
        with pytest.raises(PermissionError, match="not assigned"):
            model_access.apply_allowed_llm_selection(
                {"profile_id": CODEX_PROFILE, "model_id": "m-sol"}
            )


def test_legacy_personal_codex_profile_does_not_overlay_the_runtime_catalog(
    as_user,
    no_grants: None,
) -> None:
    _write_legacy_personal_codex(as_user, "u_alice")
    shared = _admin_catalog([{"id": SHARED_PROFILE, "models": []}])

    with as_user("u_alice"):
        assert personal_models.merge_personal_llm_profiles(shared) is shared
        assert shared["services"]["llm"]["profiles"] == [{"id": SHARED_PROFILE, "models": []}]


def test_legacy_personal_codex_profile_does_not_hide_a_valid_admin_grant(
    as_user,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = {
        "id": SHARED_PROFILE,
        "name": "Family LLM",
        "binding": "openai",
        "models": [{"id": "m-team", "name": "Team model", "model": "gpt-4o"}],
    }
    monkeypatch.setattr(model_access, "admin_catalog", lambda: _admin_catalog([shared]))
    monkeypatch.setattr(
        model_access,
        "load_grant",
        lambda _uid=None: {
            "models": {"llm": [{"profile_id": SHARED_PROFILE, "model_ids": ["m-team"]}]}
        },
    )
    _write_legacy_personal_codex(as_user, "u_alice")

    with as_user("u_alice"):
        rows = model_access.redacted_model_access()["llm"]

    assert [(row["model_id"], row["source"]) for row in rows] == [("m-team", "admin")]


def test_reading_model_options_does_not_create_a_personal_catalog(
    as_user,
    monkeypatch: pytest.MonkeyPatch,
    no_grants: None,
) -> None:
    monkeypatch.setattr(model_access, "admin_catalog", lambda: _admin_catalog([]))

    with as_user("u_alice"):
        catalog_path = personal_models.owner_catalog_service().path
        assert model_access.redacted_model_access()["llm"] == []
        assert not catalog_path.exists()
