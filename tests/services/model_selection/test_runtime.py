"""Execution-time authorisation for scoped LLM configurations."""

from __future__ import annotations

from pathlib import Path

import pytest

from deeptutor.multi_user import model_access
from deeptutor.multi_user.context import reset_current_user, set_current_user
from deeptutor.multi_user.models import CurrentUser, UserScope
from deeptutor.services.config.provider_runtime import ResolvedLLMConfig
from deeptutor.services.model_selection import runtime


def _learner(tmp_path: Path) -> CurrentUser:
    return CurrentUser(
        id="learner-1",
        username="learner@example.com",
        role="user",
        scope=UserScope(kind="user", user_id="learner-1", root=tmp_path / "learner-1"),
    )


def test_revoked_explicit_selection_never_uses_global_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token = set_current_user(_learner(tmp_path))
    try:

        def revoked(_selection):
            raise PermissionError("This model is not assigned to your account.")

        monkeypatch.setattr(model_access, "apply_allowed_llm_selection", revoked)
        monkeypatch.setattr(
            runtime.llm_config_module,
            "get_llm_config",
            lambda: (_ for _ in ()).throw(AssertionError("global config must not be used")),
        )

        with pytest.raises(PermissionError, match="not assigned"):
            runtime.resolve_llm_config_for_selection({"profile_id": "p", "model_id": "m"})
    finally:
        reset_current_user(token)


@pytest.mark.parametrize("selection", [None, {}], ids=["missing", "empty"])
def test_unassigned_learner_with_no_selection_never_uses_global_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, selection: object
) -> None:
    token = set_current_user(_learner(tmp_path))
    try:
        monkeypatch.setattr(model_access, "apply_allowed_llm_selection", lambda _selection: None)
        monkeypatch.setattr(model_access, "default_allowed_llm_selection", lambda: None)
        monkeypatch.setattr(
            runtime.llm_config_module,
            "get_llm_config",
            lambda: (_ for _ in ()).throw(AssertionError("global config must not be used")),
        )

        with pytest.raises(PermissionError, match="No LLM model is assigned"):
            runtime.resolve_llm_config_for_selection(selection)
    finally:
        reset_current_user(token)


def test_currently_granted_selection_is_resolved_after_revalidation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = {"profile_id": "cursor", "model_id": "grok-high"}
    seen: list[dict[str, str]] = []
    token = set_current_user(_learner(tmp_path))
    try:
        monkeypatch.setattr(
            model_access,
            "apply_allowed_llm_selection",
            lambda candidate: candidate,
        )
        monkeypatch.setattr(
            runtime,
            "resolve_llm_runtime_config",
            lambda *, llm_selection: (
                seen.append(llm_selection)
                or ResolvedLLMConfig(
                    model="grok-4.6-high",
                    provider_name="cursor_subscription",
                    provider_mode="oauth",
                    binding="cursor_subscription",
                )
            ),
        )

        config = runtime.resolve_llm_config_for_selection(selection)
    finally:
        reset_current_user(token)

    assert seen == [selection]
    assert config.model == "grok-4.6-high"
    assert config.binding == "cursor_subscription"


def test_scoped_human_authority_survives_nested_synthetic_workspace_and_revocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = {"profile_id": "cursor-profile", "model_id": "cursor-model"}
    catalogue = {
        model_access.DEPLOYMENT_OWNER_ID_FIELD: "u_owner",
        "services": {
            "llm": {
                "profiles": [
                    {
                        "id": selection["profile_id"],
                        "binding": "cursor_subscription",
                        "owner_bound": True,
                        "models": [
                            {
                                "id": selection["model_id"],
                                "model": "cursor-grok-4.6-high",
                            }
                        ],
                    }
                ]
            }
        },
    }
    grant_rows = [
        {
            "profile_id": selection["profile_id"],
            "model_ids": [selection["model_id"]],
            model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD: "u_owner",
        }
    ]
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalogue)
    monkeypatch.setattr(
        model_access,
        "load_grant",
        lambda _user_id: {"models": {"llm": list(grant_rows)}},
    )
    monkeypatch.setattr(
        runtime,
        "resolve_llm_runtime_config",
        lambda *, llm_selection: ResolvedLLMConfig(
            model="cursor-grok-4.6-high",
            provider_name="cursor_subscription",
            provider_mode="subscription",
            binding="cursor_subscription",
            profile_id=llm_selection["profile_id"],
            model_id=llm_selection["model_id"],
        ),
    )
    human_token = set_current_user(_learner(tmp_path))
    scope_token = None
    try:
        _config, scope_token = runtime.activate_llm_selection(selection)
        synthetic = CurrentUser(
            id="partner_ada",
            username="Ada",
            role="user",
            scope=UserScope(kind="user", user_id="partner_ada", root=tmp_path / "partner"),
        )
        synthetic_token = set_current_user(synthetic)
        try:
            model_access.require_deployment_owner_binding(
                "cursor_subscription",
                model="cursor-grok-4.6-high",
                profile_id=selection["profile_id"],
                model_id=selection["model_id"],
            )
            grant_rows.clear()
            with pytest.raises(PermissionError, match="deployment owner"):
                model_access.require_deployment_owner_binding(
                    "cursor_subscription",
                    model="cursor-grok-4.6-high",
                    profile_id=selection["profile_id"],
                    model_id=selection["model_id"],
                )
        finally:
            reset_current_user(synthetic_token)
    finally:
        runtime.reset_llm_selection(scope_token)
        reset_current_user(human_token)
