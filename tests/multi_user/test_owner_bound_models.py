"""Owner-bound profiles are never lent to other accounts through grants.

Codex, Cursor and Grok subscription bindings authenticate one adult's personal
plan rather than a billable end-user API key. Granting any of those profiles to
other users would run a deployment on that individual's subscription.
Administrators still use their own sign-in: they resolve models straight from
the catalog and never pass through the grant view tested here.
"""

from types import SimpleNamespace

import pytest

from deeptutor.multi_user import model_access
from deeptutor.multi_user.context import reset_current_user, set_current_user
from deeptutor.multi_user.models import CurrentUser, UserScope

CODEX_PROFILE = "llm-profile-openai-codex-managed"
LOCAL_PROFILE = "llm-profile-ollama"
CURSOR_PROFILE = "llm-profile-cursor-subscription"
GROK_PROFILE = "llm-profile-grok-subscription"


def make_user(tmp_path):
    return CurrentUser(
        id="u_alice",
        username="alice",
        role="user",
        scope=UserScope(kind="user", user_id="u_alice", root=tmp_path / "u_alice"),
    )


def make_admin(tmp_path, user_id: str) -> CurrentUser:
    return CurrentUser(
        id=user_id,
        username=f"{user_id}@example.com",
        role="admin",
        scope=UserScope(kind="admin", user_id=user_id, root=tmp_path / "admin"),
    )


def _catalog(*, owner_bound: bool) -> dict:
    profile: dict = {
        "id": CODEX_PROFILE,
        "name": "OpenAI Codex",
        "models": [{"id": "m-sol", "name": "GPT-5.6-Sol", "model": "gpt-5.6-sol"}],
    }
    if owner_bound:
        profile["owner_bound"] = True
    return {"services": {"llm": {"profiles": [profile]}}}


def _grant(_user_id=None) -> dict:
    return {"models": {"llm": [{"profile_id": CODEX_PROFILE, "model_ids": ["m-sol"]}]}}


def _deployment_catalog() -> dict:
    return {
        "services": {
            "llm": {
                "active_profile_id": CURSOR_PROFILE,
                "active_model_id": "m-cursor-grok-high",
                "profiles": [
                    {
                        "id": LOCAL_PROFILE,
                        "name": "Local Qwen",
                        "binding": "ollama",
                        "models": [
                            {
                                "id": "m-qwen",
                                "name": "Qwen 3.5 4B",
                                "model": "qwen3.5:4b",
                            }
                        ],
                    },
                    {
                        "id": CURSOR_PROFILE,
                        "name": "Cursor Ultra",
                        "binding": "cursor_subscription",
                        "models": [
                            {
                                "id": "m-cursor-grok-high",
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
                                "id": "m-grok-high",
                                "name": "Grok 4.6 High",
                                "model": "grok-4.6-high",
                            }
                        ],
                    },
                ],
            }
        }
    }


def _stored_admins() -> dict:
    # The insertion order is the deployment-owner contract: the first account
    # is the original administrator; later admins do not inherit its logins.
    return {
        "owner@example.com": {"id": "u_owner", "role": "admin"},
        "second@example.com": {"id": "u_second", "role": "admin"},
    }


def test_owner_bound_profile_is_withheld_from_granted_users(tmp_path, monkeypatch):
    monkeypatch.setattr(model_access, "admin_catalog", lambda: _catalog(owner_bound=True))
    monkeypatch.setattr(model_access, "load_grant", _grant)
    token = set_current_user(make_user(tmp_path))
    try:
        assert model_access.redacted_model_access()["llm"] == []
        assert model_access.allowed_llm_options()["options"] == []
        assert model_access.has_capability_access("llm") is False
        with pytest.raises(PermissionError):
            model_access.apply_allowed_llm_selection(
                {"profile_id": CODEX_PROFILE, "model_id": "m-sol"}
            )
    finally:
        reset_current_user(token)


def test_owner_bound_profile_is_not_offered_as_assignable(tmp_path, monkeypatch):
    """Admins must not be shown a grant the server would silently discard."""
    from deeptutor.multi_user import router as multi_user_router

    monkeypatch.setattr(
        multi_user_router,
        "ModelCatalogService",
        lambda path=None: SimpleNamespace(load=lambda: _catalog(owner_bound=True)),
    )
    monkeypatch.setattr(
        multi_user_router,
        "get_admin_path_service",
        lambda: SimpleNamespace(get_settings_file=lambda _name: tmp_path / "catalog.json"),
    )

    assert multi_user_router._admin_catalog_summary() == {"llm": []}


def test_ordinary_shared_profiles_stay_grantable(tmp_path, monkeypatch):
    """The filter has to stay narrow: an API-key profile is still shareable."""
    monkeypatch.setattr(model_access, "admin_catalog", lambda: _catalog(owner_bound=False))
    monkeypatch.setattr(model_access, "load_grant", _grant)
    token = set_current_user(make_user(tmp_path))
    try:
        granted = model_access.redacted_model_access()["llm"]
        assert [item["model_id"] for item in granted] == ["m-sol"]
        assert model_access.has_capability_access("llm") is True
        assert model_access.apply_allowed_llm_selection(
            {"profile_id": CODEX_PROFILE, "model_id": "m-sol"}
        ) == {"profile_id": CODEX_PROFILE, "model_id": "m-sol"}
    finally:
        reset_current_user(token)


def test_a_codebuddy_profile_is_owner_bound_by_its_binding() -> None:
    """CodeBuddy reads the operator's own IDE-plugin login on this host.

    Codex stamps ``owner_bound`` onto the managed profile it publishes, but a
    CodeBuddy profile is created by hand in the settings editor and has nowhere
    to acquire the flag — so without this the administrator's own subscription
    would be grantable to every account on the deployment.
    """
    assert model_access.is_owner_bound({"binding": "codebuddy"}) is True
    assert model_access.is_owner_bound({"binding": "CodeBuddy"}) is True
    # An ordinary team key stays grantable.
    assert model_access.is_owner_bound({"binding": "openai"}) is False
    # The explicit flag still wins for anything else that sets it.
    assert model_access.is_owner_bound({"binding": "openai", "owner_bound": True}) is True


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
def test_personal_subscription_profiles_are_owner_bound_by_binding(binding: str) -> None:
    """Hand-authored profiles cannot bypass the cross-account grant boundary."""
    assert model_access.is_owner_bound({"binding": binding}) is True
    assert model_access.is_owner_bound({"binding": binding.upper()}) is True


@pytest.mark.parametrize(
    "binding",
    ["openai-codex", "CursorSubscription", "grok-subscription"],
)
def test_subscription_aliases_are_withheld_from_existing_grants(
    binding: str, tmp_path, monkeypatch
) -> None:
    catalog = _catalog(owner_bound=False)
    catalog["services"]["llm"]["profiles"][0]["binding"] = binding
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_grant", _grant)
    token = set_current_user(make_user(tmp_path))
    try:
        assert model_access.redacted_model_access()["llm"] == []
    finally:
        reset_current_user(token)


def test_deployment_owner_admin_can_see_and_select_subscription_models(
    tmp_path, monkeypatch
) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    token = set_current_user(make_admin(tmp_path, "u_owner"))
    try:
        result = model_access.allowed_llm_options()

        assert result["active"] == {
            "profile_id": CURSOR_PROFILE,
            "model_id": "m-cursor-grok-high",
        }
        assert {item["profile_id"] for item in result["options"]} == {
            LOCAL_PROFILE,
            CURSOR_PROFILE,
            GROK_PROFILE,
        }
        for selection in (
            {"profile_id": LOCAL_PROFILE, "model_id": "m-qwen"},
            {"profile_id": CURSOR_PROFILE, "model_id": "m-cursor-grok-high"},
            {"profile_id": GROK_PROFILE, "model_id": "m-grok-high"},
        ):
            assert model_access.apply_allowed_llm_selection(selection) == selection
    finally:
        reset_current_user(token)


def test_later_admin_sees_only_local_models_and_cannot_use_owner_subscriptions(
    tmp_path, monkeypatch
) -> None:
    catalog = _deployment_catalog()
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    token = set_current_user(make_admin(tmp_path, "u_second"))
    try:
        result = model_access.allowed_llm_options()

        assert result["active"] is None
        assert [item["profile_id"] for item in result["options"]] == [LOCAL_PROFILE]

        local_selection = {"profile_id": LOCAL_PROFILE, "model_id": "m-qwen"}
        assert model_access.apply_allowed_llm_selection(local_selection) == local_selection

        for subscription_selection in (
            {"profile_id": CURSOR_PROFILE, "model_id": "m-cursor-grok-high"},
            {"profile_id": GROK_PROFILE, "model_id": "m-grok-high"},
        ):
            with pytest.raises(PermissionError, match="deployment owner"):
                model_access.apply_allowed_llm_selection(subscription_selection)

        # Omitting a per-request selection would otherwise fall through to the
        # deployment's active Cursor model, so that path must be rejected too.
        with pytest.raises(PermissionError, match="deployment owner"):
            model_access.apply_allowed_llm_selection(None)
    finally:
        reset_current_user(token)


@pytest.mark.parametrize(
    "binding",
    [
        "cursor_subscription",
        "cursor-subscription",
        "CursorSubscription",
        "grok_subscription",
        "grok-subscription",
        "GrokSubscription",
    ],
)
def test_later_admin_cannot_bypass_subscription_owner_check_with_aliases(
    binding: str, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(model_access, "load_users", _stored_admins)
    token = set_current_user(make_admin(tmp_path, "u_second"))
    try:
        with pytest.raises(PermissionError, match="deployment owner"):
            model_access.require_deployment_owner_binding(binding)
        # The boundary stays narrow: deployment-local models remain usable.
        model_access.require_deployment_owner_binding("ollama")
    finally:
        reset_current_user(token)


def test_durable_catalog_owner_does_not_transfer_when_original_account_is_removed(
    tmp_path, monkeypatch
) -> None:
    catalog = _deployment_catalog()
    catalog[model_access.DEPLOYMENT_OWNER_ID_FIELD] = "u_owner"
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
    catalog[model_access.DEPLOYMENT_OWNER_ID_FIELD] = "u_owner"
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
    catalog[model_access.DEPLOYMENT_OWNER_ID_FIELD] = "pb_owner"
    monkeypatch.setattr(model_access, "admin_catalog", lambda: catalog)
    monkeypatch.setattr(model_access, "load_users", lambda: {})
    monkeypatch.setattr(auth_service, "POCKETBASE_ENABLED", True)
    token = set_current_user(make_admin(tmp_path, "pb_owner"))
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
    """PocketBase admins have no corresponding local users.json record."""
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
    """A legacy protected profile without an owner marker fails closed."""
    catalog = _deployment_catalog()
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
    catalog = _deployment_catalog()
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
