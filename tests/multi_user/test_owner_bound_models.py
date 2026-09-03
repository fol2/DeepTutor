"""Family grants for deployment-owner subscription models.

The deployment owner keeps control of credentials and provider setup, but may
lend an exact Codex, Cursor or Grok model to an ordinary account. Grants carry
logical profile/model ids only; stale, revoked or ambiguous grants fail closed.
"""

from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from deeptutor.multi_user import model_access
from deeptutor.multi_user.context import reset_current_user, set_current_user
from deeptutor.multi_user.models import CurrentUser, UserScope

CODEX_PROFILE = "llm-profile-openai-codex-managed"
CURSOR_PROFILE = "llm-profile-cursor-subscription"
GROK_PROFILE = "llm-profile-grok-subscription"
CODEBUDDY_PROFILE = "llm-profile-codebuddy"
PRIVATE_PROFILE = "llm-profile-private-owner"
LOCAL_PROFILE = "llm-profile-ollama"

SUBSCRIPTION_CASES = (
    (CODEX_PROFILE, "openai_codex", "m-luna", "gpt-5.6-luna"),
    (CURSOR_PROFILE, "cursor_subscription", "m-cursor-grok-high", "cursor-grok-4.6-high"),
    (GROK_PROFILE, "grok_subscription", "m-grok-high", "grok-4.6-high"),
)


def make_user(tmp_path, user_id: str = "u_alice") -> CurrentUser:
    return CurrentUser(
        id=user_id,
        username=f"{user_id}@example.com",
        role="user",
        scope=UserScope(kind="user", user_id=user_id, root=tmp_path / user_id),
    )


def make_admin(tmp_path, user_id: str) -> CurrentUser:
    return CurrentUser(
        id=user_id,
        username=f"{user_id}@example.com",
        role="admin",
        scope=UserScope(kind="admin", user_id=user_id, root=tmp_path / "admin"),
    )


def _profile(profile_id: str, binding: str, model_id: str, model: str) -> dict:
    model_payload = {"id": model_id, "name": model, "model": model}
    if binding == "openai_codex" and model == "gpt-5.6-luna":
        model_payload["reasoning_effort"] = "max"
    return {
        "id": profile_id,
        "name": profile_id,
        "binding": binding,
        "owner_bound": True,
        # Deliberately include credential material to prove neither grant view
        # nor assignable-resource summaries copy it to another account.
        "api_key": "OWNER_SECRET",
        "credential_path": "/owner/private/credentials.json",
        "models": [model_payload],
    }


def _deployment_catalog(*, durable_owner: bool = True) -> dict:
    catalog = {
        "services": {
            "llm": {
                "active_profile_id": CURSOR_PROFILE,
                "active_model_id": "m-cursor-grok-high",
                "profiles": [
                    {
                        "id": LOCAL_PROFILE,
                        "name": "Local Qwen",
                        "binding": "ollama",
                        "models": [{"id": "m-qwen", "name": "Qwen", "model": "qwen3.5:4b"}],
                    },
                    *[_profile(*case) for case in SUBSCRIPTION_CASES],
                    _profile(CODEBUDDY_PROFILE, "codebuddy", "m-codebuddy", "claude-sonnet"),
                    _profile(PRIVATE_PROFILE, "openai", "m-private", "private-model"),
                ],
            }
        }
    }
    if durable_owner:
        catalog[model_access.DEPLOYMENT_OWNER_ID_FIELD] = "u_owner"
    return catalog


def _grant(profile_id: str, model_id: str, *, issued: bool = True) -> dict:
    item = {"profile_id": profile_id, "model_ids": [model_id]}
    if issued:
        item[model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD] = "u_owner"
    return {"models": {"llm": [item]}}


def _stored_admins() -> dict:
    return {
        "owner@example.com": {"id": "u_owner", "role": "admin"},
        "second@example.com": {"id": "u_second", "role": "admin"},
    }


@pytest.mark.parametrize("profile_id,binding,model_id,runtime_model", SUBSCRIPTION_CASES)
def test_exact_subscription_grant_is_available_without_exposing_credentials(
    profile_id: str,
    binding: str,
    model_id: str,
    runtime_model: str,
    tmp_path,
    monkeypatch,
) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_grant", lambda _uid=None: _grant(profile_id, model_id))
    token = set_current_user(make_user(tmp_path))
    try:
        access = model_access.redacted_model_access()["llm"]
        assert access == [
            {
                "profile_id": profile_id,
                "model_id": model_id,
                "name": runtime_model,
                "model": runtime_model,
                "source": "admin",
                "available": True,
            }
        ]
        assert "OWNER_SECRET" not in repr(access)
        assert "credential" not in repr(access).lower()
        assert model_access.allowed_llm_options()["options"][0]["model"] == runtime_model
        assert model_access.has_capability_access("llm") is True
        selection = {"profile_id": profile_id, "model_id": model_id}
        expected_selection = dict(selection)
        if binding == "openai_codex":
            expected_selection["reasoning_effort"] = "max"
        assert model_access.apply_allowed_llm_selection(selection) == expected_selection
        model_access.require_deployment_owner_binding(
            binding,
            model=runtime_model,
            profile_id=profile_id,
            model_id=model_id,
            reasoning_effort="max" if binding == "openai_codex" else None,
        )
    finally:
        reset_current_user(token)


@pytest.mark.parametrize("profile_id,binding,model_id,runtime_model", SUBSCRIPTION_CASES)
def test_subscription_grant_without_durable_owner_marker_stays_private(
    profile_id: str,
    binding: str,
    model_id: str,
    runtime_model: str,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        model_access, "admin_catalog", lambda: _deployment_catalog(durable_owner=False)
    )
    monkeypatch.setattr(model_access, "load_grant", lambda _uid=None: _grant(profile_id, model_id))
    token = set_current_user(make_user(tmp_path))
    try:
        assert model_access.redacted_model_access()["llm"] == []
        assert model_access.allowed_llm_options()["options"] == []
        with pytest.raises(PermissionError, match="deployment owner"):
            model_access.require_deployment_owner_binding(
                binding,
                model=runtime_model,
                profile_id=profile_id,
                model_id=model_id,
                reasoning_effort="max" if binding == "openai_codex" else None,
            )
    finally:
        reset_current_user(token)


@pytest.mark.parametrize("profile_id,binding,model_id,runtime_model", SUBSCRIPTION_CASES)
def test_revoked_or_wrong_subscription_model_is_denied_at_runtime(
    profile_id: str,
    binding: str,
    model_id: str,
    runtime_model: str,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_access, "admin_catalog", _deployment_catalog)
    token = set_current_user(make_user(tmp_path))
    try:
        monkeypatch.setattr(
            model_access, "load_grant", lambda _uid=None: _grant(profile_id, model_id)
        )
        model_access.require_deployment_owner_binding(
            binding,
            model=runtime_model,
            profile_id=profile_id,
            model_id=model_id,
            reasoning_effort="max" if binding == "openai_codex" else None,
        )

        with pytest.raises(PermissionError, match="deployment owner"):
            model_access.require_deployment_owner_binding(
                binding,
                model=f"{runtime_model}-wrong",
                profile_id=profile_id,
                model_id=model_id,
                reasoning_effort="max" if binding == "openai_codex" else None,
            )

        monkeypatch.setattr(model_access, "load_grant", lambda _uid=None: {"models": {"llm": []}})
        with pytest.raises(PermissionError, match="deployment owner"):
            model_access.require_deployment_owner_binding(
                binding,
                model=runtime_model,
                profile_id=profile_id,
                model_id=model_id,
                reasoning_effort="max" if binding == "openai_codex" else None,
            )
    finally:
        reset_current_user(token)


@pytest.mark.parametrize(
    "profile_id,model_id",
    [(CODEBUDDY_PROFILE, "m-codebuddy"), (PRIVATE_PROFILE, "m-private")],
)
def test_non_grantable_owner_bound_profiles_are_hidden_and_rejected(
    profile_id: str, model_id: str, tmp_path, monkeypatch
) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    monkeypatch.setattr(model_access, "load_grant", lambda _uid=None: _grant(profile_id, model_id))

    user_token = set_current_user(make_user(tmp_path))
    try:
        assert model_access.redacted_model_access()["llm"] == []
    finally:
        reset_current_user(user_token)

    owner_token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        with pytest.raises(PermissionError, match="cannot be assigned"):
            model_access.validate_assignable_model_grants(_grant(profile_id, model_id))
    finally:
        reset_current_user(owner_token)


def test_later_admin_neither_sees_nor_mints_subscription_grants(tmp_path, monkeypatch) -> None:
    from deeptutor.multi_user import router as multi_user_router

    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    monkeypatch.setattr(
        multi_user_router,
        "ModelCatalogService",
        lambda path=None: SimpleNamespace(load=lambda: catalog),
    )
    monkeypatch.setattr(
        multi_user_router,
        "get_admin_path_service",
        lambda: SimpleNamespace(get_settings_file=lambda _name: tmp_path / "catalog.json"),
    )
    token = set_current_user(make_admin(tmp_path, "u_second"))
    try:
        summary = multi_user_router._admin_catalog_summary()["llm"]
        assert [item["profile_id"] for item in summary] == [LOCAL_PROFILE]
        for profile_id, _binding, model_id, _runtime_model in SUBSCRIPTION_CASES:
            with pytest.raises(PermissionError, match="deployment owner"):
                model_access.validate_assignable_model_grants(_grant(profile_id, model_id))
    finally:
        reset_current_user(token)


def test_only_owner_preparation_stamps_subscription_provenance(tmp_path, monkeypatch) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    payload = _grant(CURSOR_PROFILE, "m-cursor-grok-high", issued=False)

    owner_token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        prepared = model_access.prepare_assignable_model_grants(payload)
    finally:
        reset_current_user(owner_token)

    row = prepared["models"]["llm"][0]
    assert row[model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD] == "u_owner"

    later_token = set_current_user(make_admin(tmp_path, "u_second"))
    try:
        with pytest.raises(PermissionError, match="deployment owner"):
            model_access.prepare_assignable_model_grants(payload)
    finally:
        reset_current_user(later_token)


def test_codex_family_grant_rejects_every_model_except_luna_max(tmp_path, monkeypatch) -> None:
    catalog = _deployment_catalog()
    codex = next(
        profile
        for profile in catalog["services"]["llm"]["profiles"]
        if profile["id"] == CODEX_PROFILE
    )
    codex["models"].append(
        {
            "id": "m-sol",
            "name": "GPT-5.6 Sol",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "max",
        }
    )
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        with pytest.raises(PermissionError, match="not approved for family use"):
            model_access.prepare_assignable_model_grants(
                _grant(CODEX_PROFILE, "m-sol", issued=False)
            )

        codex["models"][0]["reasoning_effort"] = "high"
        with pytest.raises(PermissionError, match="not approved for family use"):
            model_access.prepare_assignable_model_grants(
                _grant(CODEX_PROFILE, "m-luna", issued=False)
            )
    finally:
        reset_current_user(token)


def test_codex_learner_selection_forces_max_and_rejects_weaker_override(
    tmp_path, monkeypatch
) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(
        model_access,
        "load_grant",
        lambda _uid=None: _grant(CODEX_PROFILE, "m-luna"),
    )
    token = set_current_user(make_user(tmp_path))
    try:
        base = {"profile_id": CODEX_PROFILE, "model_id": "m-luna"}
        assert model_access.apply_allowed_llm_selection(base) == {
            **base,
            "reasoning_effort": "max",
        }
        assert model_access.apply_allowed_llm_selection({**base, "reasoning_effort": "max"}) == {
            **base,
            "reasoning_effort": "max",
        }
        with pytest.raises(PermissionError, match="fixed reasoning setting"):
            model_access.apply_allowed_llm_selection({**base, "reasoning_effort": "high"})
        model_access.require_deployment_owner_binding(
            "openai_codex",
            model="gpt-5.6-luna",
            profile_id=CODEX_PROFILE,
            model_id="m-luna",
            reasoning_effort="max",
        )
        for effort in (None, "high"):
            with pytest.raises(PermissionError, match="deployment owner"):
                model_access.require_deployment_owner_binding(
                    "openai_codex",
                    model="gpt-5.6-luna",
                    profile_id=CODEX_PROFILE,
                    model_id="m-luna",
                    reasoning_effort=effort,
                )
    finally:
        reset_current_user(token)


def test_unmarked_preupgrade_subscription_grant_never_wakes_up(tmp_path, monkeypatch) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(
        model_access,
        "load_grant",
        lambda _uid=None: _grant(CURSOR_PROFILE, "m-cursor-grok-high", issued=False),
    )
    token = set_current_user(make_user(tmp_path))
    try:
        assert model_access.redacted_model_access()["llm"] == []
        with pytest.raises(PermissionError, match="deployment owner"):
            model_access.require_deployment_owner_binding(
                "cursor_subscription",
                model="cursor-grok-4.6-high",
                profile_id=CURSOR_PROFILE,
                model_id="m-cursor-grok-high",
            )
    finally:
        reset_current_user(token)


def test_unknown_grant_cannot_be_preseeded_before_profile_creation(tmp_path, monkeypatch) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        with pytest.raises(ValueError, match="Unknown LLM profile"):
            model_access.prepare_assignable_model_grants(
                _grant("llm-profile-future-subscription", "future-model", issued=False)
            )
    finally:
        reset_current_user(token)


def test_grant_for_one_identical_binding_profile_does_not_authorise_another(
    tmp_path, monkeypatch
) -> None:
    catalog = _deployment_catalog()
    twin = _profile(
        "llm-profile-cursor-subscription-twin",
        "cursor_subscription",
        "m-cursor-grok-high-twin",
        "cursor-grok-4.6-high",
    )
    catalog["services"]["llm"]["profiles"].append(twin)
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(
        model_access,
        "load_grant",
        lambda _uid=None: _grant(CURSOR_PROFILE, "m-cursor-grok-high"),
    )
    token = set_current_user(make_user(tmp_path))
    try:
        model_access.require_deployment_owner_binding(
            "cursor_subscription",
            model="cursor-grok-4.6-high",
            profile_id=CURSOR_PROFILE,
            model_id="m-cursor-grok-high",
        )
        with pytest.raises(PermissionError, match="deployment owner"):
            model_access.require_deployment_owner_binding(
                "cursor_subscription",
                model="cursor-grok-4.6-high",
                profile_id=twin["id"],
                model_id=twin["models"][0]["id"],
            )
    finally:
        reset_current_user(token)


def test_later_admin_update_preserves_hidden_protected_rows(tmp_path, monkeypatch) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    existing = {
        "models": {
            "llm": [
                _grant(CURSOR_PROFILE, "m-cursor-grok-high")["models"]["llm"][0],
                {"profile_id": "deleted-profile", "model_ids": ["deleted-model"]},
            ]
        },
        "knowledge_bases": [{"resource_id": "admin:kb:old"}],
    }
    incoming = {
        "models": {"llm": [{"profile_id": LOCAL_PROFILE, "model_ids": ["m-qwen"]}]},
        "knowledge_bases": [{"resource_id": "admin:kb:new"}],
        "enabled_tools": [],
    }
    token = set_current_user(make_admin(tmp_path, "u_second"))
    try:
        assert model_access.admin_visible_grant(existing)["models"]["llm"] == []
        prepared = model_access.prepare_assignable_model_grants(incoming, existing=existing)
        visible = model_access.admin_visible_grant(prepared)
    finally:
        reset_current_user(token)

    assert prepared["models"]["llm"] == [
        incoming["models"]["llm"][0],
        existing["models"]["llm"][0],
        existing["models"]["llm"][1],
    ]
    assert visible["models"]["llm"] == [incoming["models"]["llm"][0]]
    assert visible["knowledge_bases"] == incoming["knowledge_bases"]
    assert model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD not in repr(visible)


def test_owner_can_revoke_subscription_by_omitting_it(tmp_path, monkeypatch) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    existing = _grant(CURSOR_PROFILE, "m-cursor-grok-high")
    token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        prepared = model_access.prepare_assignable_model_grants(
            {"models": {"llm": []}}, existing=existing
        )
    finally:
        reset_current_user(token)
    assert prepared["models"]["llm"] == []


@pytest.mark.asyncio
async def test_later_admin_grant_api_hides_and_preserves_owner_subscription(
    tmp_path, monkeypatch
) -> None:
    from deeptutor.multi_user import router as multi_user_router

    catalog = _deployment_catalog()
    existing = {
        **_grant(CURSOR_PROFILE, "m-cursor-grok-high"),
        "knowledge_bases": [{"resource_id": "admin:kb:old"}],
    }
    saved: list[dict] = []
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    monkeypatch.setattr(
        multi_user_router,
        "get_user_by_id",
        lambda user_id: ("child@example.com", {"id": user_id, "role": "user"}),
    )
    monkeypatch.setattr(multi_user_router, "load_grant", lambda _user_id: existing)

    def update_grant(_user_id, updater):
        grant = updater(existing)
        saved.append(grant)
        return grant

    monkeypatch.setattr(multi_user_router, "update_grant", update_grant)
    monkeypatch.setattr(multi_user_router, "log_admin_action", lambda *args, **kwargs: None)

    token = set_current_user(make_admin(tmp_path, "u_second"))
    try:
        before = await multi_user_router.get_user_grants("u_child", None)
        after = await multi_user_router.put_user_grants(
            "u_child",
            multi_user_router.GrantPayload(
                grant={
                    "models": {"llm": [{"profile_id": LOCAL_PROFILE, "model_ids": ["m-qwen"]}]},
                    "knowledge_bases": [{"resource_id": "admin:kb:new"}],
                    "enabled_tools": [],
                }
            ),
            None,
        )
    finally:
        reset_current_user(token)

    assert before["grant"]["models"]["llm"] == []
    assert after["grant"]["models"]["llm"] == [
        {"profile_id": LOCAL_PROFILE, "model_ids": ["m-qwen"]}
    ]
    assert saved[0]["models"]["llm"][1] == existing["models"]["llm"][0]


@pytest.mark.asyncio
async def test_grant_api_rejects_unknown_future_profile(tmp_path, monkeypatch) -> None:
    from deeptutor.multi_user import router as multi_user_router

    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    monkeypatch.setattr(
        multi_user_router,
        "get_user_by_id",
        lambda user_id: ("child@example.com", {"id": user_id, "role": "user"}),
    )
    monkeypatch.setattr(
        multi_user_router,
        "load_grant",
        lambda _user_id: {"models": {"llm": []}},
    )
    token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        with pytest.raises(HTTPException) as exc_info:
            await multi_user_router.put_user_grants(
                "u_child",
                multi_user_router.GrantPayload(
                    grant={
                        "models": {
                            "llm": [
                                {
                                    "profile_id": "llm-profile-future-subscription",
                                    "model_ids": ["future-model"],
                                }
                            ]
                        }
                    }
                ),
                None,
            )
    finally:
        reset_current_user(token)

    assert exc_info.value.status_code == 400
    assert "Unknown LLM profile" in str(exc_info.value.detail)


def test_deployment_owner_sees_only_safe_grantable_resource_summaries(
    tmp_path, monkeypatch
) -> None:
    from deeptutor.multi_user import router as multi_user_router

    catalog = _deployment_catalog()
    codex = next(
        profile
        for profile in catalog["services"]["llm"]["profiles"]
        if profile["id"] == CODEX_PROFILE
    )
    codex["models"].append({"id": "m-sol", "name": "GPT-5.6 Sol", "model": "gpt-5.6-sol"})
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    monkeypatch.setattr(
        multi_user_router,
        "ModelCatalogService",
        lambda path=None: SimpleNamespace(load=lambda: catalog),
    )
    monkeypatch.setattr(
        multi_user_router,
        "get_admin_path_service",
        lambda: SimpleNamespace(get_settings_file=lambda _name: tmp_path / "catalog.json"),
    )
    token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        summary = multi_user_router._admin_catalog_summary()["llm"]
        assert {item["profile_id"] for item in summary} == {
            LOCAL_PROFILE,
            CODEX_PROFILE,
            CURSOR_PROFILE,
            GROK_PROFILE,
        }
        codex_summary = next(item for item in summary if item["profile_id"] == CODEX_PROFILE)
        assert codex_summary["models"] == [
            {"model_id": "m-luna", "name": "gpt-5.6-luna", "model": "gpt-5.6-luna"}
        ]
        rendered = repr(summary)
        assert "OWNER_SECRET" not in rendered
        assert "api_key" not in rendered
        assert "credential_path" not in rendered
        assert CODEBUDDY_PROFILE not in rendered
        assert PRIVATE_PROFILE not in rendered
    finally:
        reset_current_user(token)


@pytest.mark.parametrize(
    "binding",
    [
        "openai_codex",
        "openai-codex",
        "OpenAICodex",
        "cursor_subscription",
        "cursor-subscription",
        "CursorSubscription",
        "grok_subscription",
        "grok-subscription",
        "GrokSubscription",
    ],
)
def test_family_subscription_aliases_are_owner_bound_and_grantable(binding: str) -> None:
    assert model_access.is_owner_bound({"binding": binding}) is True
    assert model_access.is_grantable_subscription_binding(binding) is True


def test_codebuddy_and_arbitrary_owner_bound_profiles_are_not_grantable() -> None:
    assert model_access.is_owner_bound({"binding": "codebuddy"}) is True
    assert model_access.is_grantable_subscription_binding("codebuddy") is False
    assert model_access.is_owner_bound({"binding": "openai", "owner_bound": True}) is True
    assert model_access.is_grantable_subscription_binding("openai") is False


def test_durable_catalog_owner_does_not_transfer_when_original_account_is_removed(
    tmp_path, monkeypatch
) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(
        model_access,
        "load_users",
        lambda: {"second@example.com": {"id": "u_second", "role": "admin"}},
    )
    token = set_current_user(make_admin(tmp_path, "u_second"))
    try:
        assert model_access.is_deployment_owner() is False
        with pytest.raises(PermissionError, match="deployment owner"):
            model_access.require_deployment_owner_binding("cursor_subscription")
    finally:
        reset_current_user(token)


@pytest.mark.parametrize(
    "stored_owner",
    [
        {"id": "u_owner", "role": "user", "disabled": False},
        {"id": "u_owner", "role": "admin", "disabled": True},
        None,
    ],
    ids=["demoted", "disabled", "deleted"],
)
def test_durable_owner_requires_current_admin_account(stored_owner, tmp_path, monkeypatch) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    users = {} if stored_owner is None else {"owner@example.com": stored_owner}
    monkeypatch.setattr(model_access, "load_users", lambda: users)
    token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        assert model_access.is_deployment_owner() is False
        with pytest.raises(PermissionError, match="deployment owner"):
            model_access.require_deployment_owner_binding("cursor_subscription")
    finally:
        reset_current_user(token)


def test_pocketbase_owner_uses_current_authenticated_role_without_local_mirror(
    tmp_path, monkeypatch
) -> None:
    from deeptutor.services import auth as auth_service

    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", lambda: {})
    monkeypatch.setattr(auth_service, "POCKETBASE_ENABLED", True)
    token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        assert model_access.is_deployment_owner() is True
    finally:
        reset_current_user(token)


def test_first_stored_admin_is_owner_even_when_a_child_record_precedes_it(
    tmp_path, monkeypatch
) -> None:
    catalog = {"services": {"llm": {"profiles": []}}}
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(
        model_access,
        "load_users",
        lambda: {
            "child@example.com": {"id": "u_child", "role": "user"},
            "owner@example.com": {"id": "u_owner", "role": "admin"},
            "second@example.com": {"id": "u_second", "role": "admin"},
        },
    )

    owner_token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        assert model_access.is_deployment_owner() is True
    finally:
        reset_current_user(owner_token)

    second_token = set_current_user(make_admin(tmp_path, "u_second"))
    try:
        assert model_access.is_deployment_owner() is False
    finally:
        reset_current_user(second_token)


def test_external_identity_admin_can_claim_an_empty_catalogue(tmp_path, monkeypatch) -> None:
    catalog = {"services": {"llm": {"profiles": []}}}
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", lambda: {})
    token = set_current_user(make_admin(tmp_path, "pb_owner"))
    try:
        assert model_access.is_deployment_owner() is True
    finally:
        reset_current_user(token)


def test_external_identity_admin_cannot_adopt_ambiguous_unbound_subscription(
    tmp_path, monkeypatch
) -> None:
    catalog = _deployment_catalog(durable_owner=False)
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", lambda: {})
    token = set_current_user(make_admin(tmp_path, "pb_admin"))
    try:
        assert model_access.is_deployment_owner() is False
    finally:
        reset_current_user(token)


@pytest.mark.parametrize("user_id", ["local-admin", "env-admin"])
def test_explicit_bootstrap_admin_is_owner_before_a_durable_binding_exists(
    user_id: str, tmp_path, monkeypatch
) -> None:
    catalog = _deployment_catalog(durable_owner=False)
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(
        model_access,
        "load_users",
        lambda: {"child@example.com": {"id": "u_child", "role": "user"}},
    )
    token = set_current_user(make_admin(tmp_path, user_id))
    try:
        assert model_access.is_deployment_owner() is True
    finally:
        reset_current_user(token)
