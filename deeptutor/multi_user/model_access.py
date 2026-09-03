"""Server-side model grant resolution and redacted model views.

Grants carry LLM assignments only (grant v2): embedding and search always
resolve from the deployment's active profiles, so per-user grants for them
were never enforced and are not stored.

Only administrator-assigned models reach an ordinary user, and
:func:`redacted_model_access` is the one place those grants are resolved.
Credentials for consumer-subscription connectors remain deployment-owner
state, while the owner may lend selected models to individual users through an
exact profile/model grant. Everything downstream — the option list, the
capability gate, selection validation, and the runtime dispatch gate — resolves
the same logical ids, so credentials never enter a user grant.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from copy import deepcopy
from typing import Any

from deeptutor.services.config.model_catalog import ModelCatalogService
from deeptutor.services.model_selection import list_llm_options
from deeptutor.services.provider_registry import canonical_provider_name

from .context import get_current_user
from .grants import load_grant
from .identity import load_users
from .models import LOCAL_ADMIN_ID, CurrentUser
from .paths import get_admin_path_service


def admin_catalog_service() -> ModelCatalogService:
    return ModelCatalogService(path=get_admin_path_service().get_settings_file("model_catalog"))


def admin_catalog() -> dict[str, Any]:
    return admin_catalog_service().load()


def _profile_by_id(catalog: dict[str, Any], service: str, profile_id: str) -> dict[str, Any] | None:
    for profile in catalog.get("services", {}).get(service, {}).get("profiles", []) or []:
        if str(profile.get("id") or "") == profile_id:
            return profile
    return None


def _model_by_id(profile: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    for model in profile.get("models", []) or []:
        if str(model.get("id") or "") == model_id:
            return model
    return None


#: Bindings whose credential is one person's own subscription login rather than
#: a billable team key. Codex stamps ``owner_bound`` onto the managed profile it
#: publishes, but a profile can also be created by hand in the settings editor —
#: a CodeBuddy profile is, and it reads the operator's own IDE-plugin session —
#: and there is nowhere for such a profile to acquire the flag. Binding is the
#: durable fact, so it decides too.
OWNER_BOUND_BINDINGS = frozenset(
    {"openai_codex", "codebuddy", "cursor_subscription", "grok_subscription"}
)
DEPLOYMENT_OWNER_BINDINGS = frozenset(
    {"openai_codex", "codebuddy", "cursor_subscription", "grok_subscription"}
)
# The deployment owner may explicitly lend these fixed, text-only subscription
# models to a family account. CodeBuddy remains operator-only because it reads
# an IDE-plugin session and was not designed as a shared tutoring backend.
GRANTABLE_SUBSCRIPTION_BINDINGS = frozenset(
    {"openai_codex", "cursor_subscription", "grok_subscription"}
)
# The family surface is intentionally narrower than each owner's provider
# catalogue.  The owner can continue to use every model exposed by their
# subscriptions, while learner grants are limited to the exact tutoring
# choices approved for this deployment.  ``reasoning_effort`` is forced onto
# the returned learner selection when present, so a saved conversation cannot
# weaken the requested Codex setting with a per-turn override.
FAMILY_SUBSCRIPTION_MODEL_POLICIES: dict[str, dict[str, str | None]] = {
    "openai_codex": {"gpt-5.6-luna": "max"},
    "cursor_subscription": {"cursor-grok-4.6-high": None},
    "grok_subscription": {"grok-4.6-high": None},
}
DEPLOYMENT_OWNER_ID_FIELD = "deployment_owner_user_id"
# Internal provenance stamped only by the deployment owner's grant endpoint.
# A catalogue marker by itself is insufficient: old/guessed logical ids must
# not spring to life when a subscription profile is added later.
SUBSCRIPTION_GRANT_ISSUER_FIELD = "subscription_grant_issued_by"
_OWNER_BOUND_COMPACT_BINDINGS = frozenset(
    binding.replace("_", "") for binding in OWNER_BOUND_BINDINGS
)
_DEPLOYMENT_OWNER_COMPACT_BINDINGS = frozenset(
    binding.replace("_", "") for binding in DEPLOYMENT_OWNER_BINDINGS
)
_GRANTABLE_SUBSCRIPTION_COMPACT_BINDINGS = frozenset(
    binding.replace("_", "") for binding in GRANTABLE_SUBSCRIPTION_BINDINGS
)
_RUNTIME_MODEL_AUTHORITY: ContextVar[CurrentUser | None] = ContextVar(
    "deeptutor_runtime_model_authority",
    default=None,
)


def set_runtime_model_authority(user: CurrentUser) -> Token[CurrentUser | None]:
    """Pin the human whose grant authorises a scoped model selection."""
    return _RUNTIME_MODEL_AUTHORITY.set(user)


def reset_runtime_model_authority(token: Token[CurrentUser | None]) -> None:
    _RUNTIME_MODEL_AUTHORITY.reset(token)


def get_runtime_model_authority() -> CurrentUser | None:
    return _RUNTIME_MODEL_AUTHORITY.get()


def _binding_keys(value: Any) -> tuple[str | None, str]:
    raw_binding = str(value or "").strip()
    canonical = canonical_provider_name(raw_binding)
    compact = "".join(character for character in raw_binding.casefold() if character.isalnum())
    return canonical, compact


def is_owner_bound(profile: dict[str, Any]) -> bool:
    """Whether a profile is tied to the identity of the operator who set it up.

    OAuth providers such as Codex authenticate one individual's plan rather than
    a billable team key, so credential and lifecycle access always stays with
    that operator. A separate allow-list decides which exact models may be lent
    through secret-free grants.
    """
    binding, compact_binding = _binding_keys(profile.get("binding") or profile.get("provider"))
    if binding in OWNER_BOUND_BINDINGS or compact_binding in _OWNER_BOUND_COMPACT_BINDINGS:
        return True
    return bool(profile.get("owner_bound"))


def is_grantable_subscription_binding(binding: Any) -> bool:
    """Whether an owner-managed binding supports explicit per-user grants."""
    canonical, compact = _binding_keys(binding)
    return not (
        canonical not in GRANTABLE_SUBSCRIPTION_BINDINGS
        and compact not in _GRANTABLE_SUBSCRIPTION_COMPACT_BINDINGS
    )


def is_grantable_subscription_profile(
    profile: dict[str, Any],
    catalog: dict[str, Any],
) -> bool:
    """Whether *profile* is durably owner-managed and may be granted.

    Requiring a durable deployment-owner marker prevents an old hand-authored
    subscription profile from becoming shareable merely because a stale grant
    references its id.
    """
    binding = profile.get("binding") or profile.get("provider")
    return bool(catalog.get(DEPLOYMENT_OWNER_ID_FIELD)) and is_grantable_subscription_binding(
        binding
    )


def _family_subscription_policy(
    profile: dict[str, Any],
    model: dict[str, Any],
) -> tuple[bool, str | None]:
    """Return whether this exact owner model is approved for family use."""
    binding, compact = _binding_keys(profile.get("binding") or profile.get("provider"))
    if binding not in FAMILY_SUBSCRIPTION_MODEL_POLICIES:
        binding = next(
            (
                candidate
                for candidate in FAMILY_SUBSCRIPTION_MODEL_POLICIES
                if candidate.replace("_", "") == compact
            ),
            None,
        )
    if binding is None:
        return False, None
    model_name = str(model.get("model") or "").strip()
    policies = FAMILY_SUBSCRIPTION_MODEL_POLICIES[binding]
    if model_name not in policies:
        return False, None
    return True, policies[model_name]


def is_grantable_subscription_model(
    profile: dict[str, Any],
    model: dict[str, Any],
    catalog: dict[str, Any],
) -> bool:
    """Whether this exact model is both owner-managed and family-approved."""
    allowed, required_effort = _family_subscription_policy(profile, model)
    if not allowed or not is_grantable_subscription_profile(profile, catalog):
        return False
    if required_effort is None:
        return True
    return str(model.get("reasoning_effort") or "").strip().lower() == required_effort


def is_owner_issued_subscription_grant(
    item: dict[str, Any],
    catalog: dict[str, Any],
) -> bool:
    """Whether *item* was explicitly issued by this catalogue's owner."""
    owner_id = str(catalog.get(DEPLOYMENT_OWNER_ID_FIELD) or "")
    return bool(owner_id) and str(item.get(SUBSCRIPTION_GRANT_ISSUER_FIELD) or "") == owner_id


def is_deployment_owner(
    user: CurrentUser | None = None,
    catalog: dict[str, Any] | None = None,
) -> bool:
    """Whether *user* owns deployment-global personal subscription state.

    Local single-user runs and the configured environment bootstrap admin are
    the operator by definition. In stored multi-user deployments, the first
    stored administrator is the initial owner candidate; records for children
    or other ordinary users may legitimately precede it. Backends such as
    PocketBase do not populate the local identity store, so an otherwise
    unclaimed catalogue may be claimed by its first authenticated admin write.
    The durable catalogue marker decides every request after that write.
    """
    actor = user or get_current_user()
    if not actor.is_admin:
        return False
    source = catalog if catalog is not None else admin_catalog()
    bound_owner_id = str(source.get(DEPLOYMENT_OWNER_ID_FIELD) or "")
    if bound_owner_id:
        if actor.id != bound_owner_id:
            return False
        if actor.id in {LOCAL_ADMIN_ID, "env-admin"}:
            return True
        users = load_users()
        matching_record = next(
            (
                record
                for record in users.values()
                if isinstance(record, dict) and str(record.get("id") or "") == actor.id
            ),
            None,
        )
        if matching_record is not None:
            return matching_record.get("role") == "admin" and not bool(
                matching_record.get("disabled")
            )
        # PocketBase identities are validated by auth_refresh on each request
        # (and re-read directly by scheduled jobs), but are not mirrored into
        # users.json. Trust that current authenticated role only in that mode.
        from deeptutor.services import auth as auth_service

        return bool(auth_service.POCKETBASE_ENABLED)
    if actor.id in {LOCAL_ADMIN_ID, "env-admin"}:
        return True
    users = load_users()
    first_admin = next(
        (
            record
            for record in users.values()
            if isinstance(record, dict)
            and record.get("role") == "admin"
            and not bool(record.get("disabled"))
        ),
        None,
    )
    if first_admin is not None:
        return str(first_admin.get("id") or "") == actor.id

    # PocketBase identities are deliberately not mirrored into users.json.
    # Before any protected profile exists, let the first authenticated admin
    # save one; _prepare_catalog_write stamps that exact actor id durably. An
    # old protected profile without a marker is ambiguous and therefore stays
    # closed rather than being exposed to every administrator.
    return not any(
        is_deployment_owner_binding(profile.get("binding") or profile.get("provider"))
        for profile in source.get("services", {}).get("llm", {}).get("profiles", [])
        if isinstance(profile, dict)
    )


def require_deployment_owner_binding(
    binding: str | None,
    user: CurrentUser | None = None,
    model: str | None = None,
    profile_id: str | None = None,
    model_id: str | None = None,
    reasoning_effort: str | None = None,
) -> None:
    """Protect owner credentials while allowing an exact granted model.

    Settings and OAuth callers do not supply a model, so they remain strictly
    deployment-owner-only. Runtime callers supply the resolved model and an
    ordinary user may proceed only when a current logical grant resolves to
    that binding/model pair in the owner-managed catalogue.
    """
    if not is_deployment_owner_binding(binding):
        return
    actor = user or get_runtime_model_authority() or get_current_user()
    catalog = admin_catalog()
    if is_deployment_owner(actor, catalog):
        return
    if not actor.is_admin and _has_granted_subscription_model(
        actor.id,
        binding,
        model,
        profile_id,
        model_id,
        reasoning_effort,
        catalog,
    ):
        return
    raise PermissionError("This subscription model is reserved for the deployment owner.")


def is_deployment_owner_binding(binding: Any) -> bool:
    """Whether *binding* consumes deployment-global personal credentials."""
    canonical, compact = _binding_keys(binding)
    return not (
        canonical not in DEPLOYMENT_OWNER_BINDINGS
        and compact not in _DEPLOYMENT_OWNER_COMPACT_BINDINGS
    )


def _admin_selection_binding(selection: dict[str, Any] | None) -> str:
    catalog = admin_catalog()
    service = catalog.get("services", {}).get("llm", {})
    profile_id = str((selection or {}).get("profile_id") or service.get("active_profile_id") or "")
    profile = _profile_by_id(catalog, "llm", profile_id)
    return str((profile or {}).get("binding") or (profile or {}).get("provider") or "")


def _has_granted_subscription_model(
    user_id: str,
    binding: Any,
    model: str | None,
    profile_id: str | None,
    model_id: str | None,
    reasoning_effort: str | None,
    catalog: dict[str, Any],
) -> bool:
    """Resolve a runtime subscription grant without exposing its credential."""
    expected_binding, expected_compact = _binding_keys(binding)
    expected_model = str(model or "").strip()
    expected_profile_id = str(profile_id or "").strip()
    expected_model_id = str(model_id or "").strip()
    if (
        not expected_model
        or not expected_profile_id
        or not expected_model_id
        or not is_grantable_subscription_binding(binding)
    ):
        return False
    grant = load_grant(user_id)
    for item in grant.get("models", {}).get("llm", []) or []:
        granted_profile_id = str(item.get("profile_id") or item.get("id") or "")
        if granted_profile_id != expected_profile_id:
            continue
        profile = _profile_by_id(catalog, "llm", granted_profile_id)
        if profile is None or not is_grantable_subscription_profile(profile, catalog):
            continue
        if not is_owner_issued_subscription_grant(item, catalog):
            continue
        actual_binding, actual_compact = _binding_keys(
            profile.get("binding") or profile.get("provider")
        )
        if actual_binding != expected_binding and actual_compact != expected_compact:
            continue
        for granted_model_id in item.get("model_ids") or []:
            if str(granted_model_id) != expected_model_id:
                continue
            candidate = _model_by_id(profile, expected_model_id)
            _allowed, required_effort = _family_subscription_policy(profile, candidate or {})
            if (
                candidate is not None
                and str(candidate.get("model") or "") == expected_model
                and is_grantable_subscription_model(profile, candidate, catalog)
                and (
                    required_effort is None
                    or str(reasoning_effort or "").strip().lower() == required_effort
                )
            ):
                return True
    return False


def _grant_llm_items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    models = payload.get("models") if isinstance(payload, dict) else None
    items = models.get("llm") if isinstance(models, dict) else None
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _validate_known_model_item(
    item: dict[str, Any],
    catalog: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    profile_id = str(item.get("profile_id") or item.get("id") or "")
    profile = _profile_by_id(catalog, "llm", profile_id)
    if profile is None:
        raise ValueError(f"Unknown LLM profile id: {profile_id or '<empty>'}")
    raw_model_ids = item.get("model_ids")
    model_ids = [str(value) for value in raw_model_ids] if isinstance(raw_model_ids, list) else []
    for model_id in model_ids:
        if _model_by_id(profile, model_id) is None:
            raise ValueError(f"Unknown LLM model id: {model_id or '<empty>'}")
    return profile, model_ids


def _is_protected_grant_item(item: dict[str, Any], catalog: dict[str, Any]) -> bool:
    """Rows a later administrator must neither see nor overwrite."""
    profile_id = str(item.get("profile_id") or item.get("id") or "")
    profile = _profile_by_id(catalog, "llm", profile_id)
    # Unknown legacy rows stay opaque. They remain unavailable at runtime and
    # cannot be resubmitted, but an unrelated admin edit must not silently
    # destroy historical state either.
    return profile is None or is_owner_bound(profile)


def prepare_assignable_model_grants(
    payload: dict[str, Any] | None,
    *,
    existing: dict[str, Any] | None = None,
    user: CurrentUser | None = None,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a grant update and stamp/preserve protected model rows.

    The deployment owner may issue known exact subscription pairs. Other
    administrators can edit ordinary resources, but protected rows are hidden
    from their payload and merged back unchanged server-side. Unknown submitted
    ids are always rejected, closing the future-profile pre-seeding path.
    """
    actor = user or get_current_user()
    source = catalog if catalog is not None else admin_catalog()
    prepared = deepcopy(payload) if isinstance(payload, dict) else {}
    models = prepared.get("models")
    if not isinstance(models, dict):
        models = {}
        prepared["models"] = models
    submitted = _grant_llm_items(prepared)
    actor_is_owner = is_deployment_owner(actor, source)
    owner_id = str(source.get(DEPLOYMENT_OWNER_ID_FIELD) or "")
    prepared_items: list[dict[str, Any]] = []
    for item in submitted:
        profile, model_ids = _validate_known_model_item(item, source)
        candidate = deepcopy(item)
        candidate.pop(SUBSCRIPTION_GRANT_ISSUER_FIELD, None)
        if is_owner_bound(profile):
            if not is_grantable_subscription_profile(profile, source):
                raise PermissionError(
                    "This owner-bound model cannot be assigned to another account."
                )
            if not model_ids or any(
                not is_grantable_subscription_model(
                    profile,
                    _model_by_id(profile, model_id) or {},
                    source,
                )
                for model_id in model_ids
            ):
                raise PermissionError(
                    "This subscription model or reasoning setting is not approved for family use."
                )
            if not actor_is_owner:
                raise PermissionError(
                    "Subscription model assignments are managed by the deployment owner."
                )
            candidate[SUBSCRIPTION_GRANT_ISSUER_FIELD] = owner_id
        prepared_items.append(candidate)

    if not actor_is_owner:
        prepared_items.extend(
            deepcopy(item)
            for item in _grant_llm_items(existing)
            if _is_protected_grant_item(item, source)
        )
    models["llm"] = prepared_items
    return prepared


def validate_assignable_model_grants(
    payload: dict[str, Any] | None,
    *,
    existing: dict[str, Any] | None = None,
    user: CurrentUser | None = None,
    catalog: dict[str, Any] | None = None,
) -> None:
    """Validate a grant update without returning its prepared representation."""
    prepare_assignable_model_grants(
        payload,
        existing=existing,
        user=user,
        catalog=catalog,
    )


def admin_visible_grant(
    grant: dict[str, Any],
    *,
    user: CurrentUser | None = None,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the editable grant view for the current administrator.

    Provenance is server-only. The owner sees current owner-issued subscription
    rows; later administrators see neither subscription nor opaque legacy rows.
    """
    actor = user or get_current_user()
    source = catalog if catalog is not None else admin_catalog()
    actor_is_owner = is_deployment_owner(actor, source)
    visible = deepcopy(grant)
    models = visible.get("models")
    if not isinstance(models, dict):
        models = {}
        visible["models"] = models
    rows: list[dict[str, Any]] = []
    for item in _grant_llm_items(grant):
        profile_id = str(item.get("profile_id") or item.get("id") or "")
        profile = _profile_by_id(source, "llm", profile_id)
        if profile is None:
            continue
        if is_owner_bound(profile):
            if not (
                actor_is_owner
                and is_grantable_subscription_profile(profile, source)
                and is_owner_issued_subscription_grant(item, source)
                and bool(item.get("model_ids"))
                and all(
                    is_grantable_subscription_model(
                        profile,
                        _model_by_id(profile, str(model_id)) or {},
                        source,
                    )
                    for model_id in item.get("model_ids") or []
                )
            ):
                continue
        candidate = deepcopy(item)
        candidate.pop(SUBSCRIPTION_GRANT_ISSUER_FIELD, None)
        rows.append(candidate)
    models["llm"] = rows
    return visible


def redacted_model_access(user_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
    user = get_current_user()
    if user_id is None:
        user_id = user.id
    grant = load_grant(user_id)
    catalog = admin_catalog()
    result: dict[str, list[dict[str, Any]]] = {"llm": []}
    for item in grant.get("models", {}).get("llm", []) or []:
        profile_id = str(item.get("profile_id") or item.get("id") or "")
        profile = _profile_by_id(catalog, "llm", profile_id)
        if (
            profile is not None
            and is_owner_bound(profile)
            and (
                not is_grantable_subscription_profile(profile, catalog)
                or not is_owner_issued_subscription_grant(item, catalog)
            )
        ):
            # Arbitrary owner-bound profiles, legacy rows and guessed logical ids
            # remain private until the deployment owner explicitly issues them.
            continue
        if not profile:
            result["llm"].append(
                {
                    "profile_id": profile_id,
                    "name": item.get("name") or profile_id or "Unavailable profile",
                    "source": "admin",
                    "available": False,
                }
            )
            continue
        for model_id in item.get("model_ids") or []:
            model = _model_by_id(profile, str(model_id))
            if (
                model is not None
                and is_owner_bound(profile)
                and not is_grantable_subscription_model(profile, model, catalog)
            ):
                continue
            result["llm"].append(
                {
                    "profile_id": profile_id,
                    "model_id": str(model_id),
                    "name": (model or {}).get("name") or str(model_id),
                    "model": (model or {}).get("model") or "",
                    "source": "admin",
                    "available": model is not None,
                }
            )
    return result


def allowed_llm_options(catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    user = get_current_user()
    if user.is_admin:
        if catalog is None:
            catalog = admin_catalog()
        result = list_llm_options(catalog)
        blocked_profile_ids = {
            str(profile.get("id") or "")
            for profile in catalog.get("services", {}).get("llm", {}).get("profiles", [])
            if isinstance(profile, dict)
            and is_deployment_owner_binding(profile.get("binding") or profile.get("provider"))
        }
        if blocked_profile_ids and not is_deployment_owner(user, catalog):
            result["options"] = [
                option
                for option in result["options"]
                if option.get("profile_id") not in blocked_profile_ids
            ]
            active = result.get("active")
            if isinstance(active, dict) and active.get("profile_id") in blocked_profile_ids:
                result["active"] = None
        return result
    options = [
        {
            "profile_id": item.get("profile_id"),
            "model_id": item.get("model_id"),
            "profile_name": item.get("name") or item.get("profile_id") or "LLM",
            "model_name": item.get("name") or item.get("model") or item.get("model_id"),
            "label": item.get("name") or item.get("model") or item.get("model_id"),
            "model": item.get("model") or "",
            "provider": "",
            "source": item.get("source") or "admin",
            "is_active_default": False,
        }
        for item in redacted_model_access(user.id).get("llm", [])
        if item.get("available")
    ]
    return {"active": None, "options": options}


def has_capability_access(capability: str, user_id: str | None = None) -> bool:
    """Whether the user has at least one usable model for ``capability``.

    Admins are never gated — they manage the catalog directly. For ordinary
    users this mirrors exactly what ``redacted_model_access`` exposes to the
    frontend, so the server-side gate and the UI lock always agree.
    """
    user = get_current_user()
    if user.is_admin:
        return True
    if user_id is None:
        user_id = user.id
    items = redacted_model_access(user_id).get(capability, []) or []
    return any(item.get("available") for item in items)


def default_allowed_llm_selection(user_id: str | None = None) -> dict[str, str] | None:
    """Return the first current exact model grant for an ordinary user."""
    user = get_current_user()
    if user.is_admin:
        return None
    target_user_id = user_id or user.id
    for item in redacted_model_access(target_user_id).get("llm", []) or []:
        if item.get("available"):
            return {
                "profile_id": str(item.get("profile_id") or ""),
                "model_id": str(item.get("model_id") or ""),
            }
    return None


def apply_allowed_llm_selection(selection: dict[str, Any] | None) -> dict[str, Any] | None:
    """Allow only admin-granted LLM profile/model selections for ordinary users."""
    user = get_current_user()
    if user.is_admin:
        require_deployment_owner_binding(_admin_selection_binding(selection), user)
        return selection
    if not selection:
        return selection
    profile_id = str(selection.get("profile_id") or "")
    model_id = str(selection.get("model_id") or "")
    for item in redacted_model_access(user.id).get("llm", []):
        if item.get("profile_id") == profile_id and item.get("model_id") == model_id:
            catalog = admin_catalog()
            profile = _profile_by_id(catalog, "llm", profile_id)
            model = _model_by_id(profile, model_id) if profile is not None else None
            if profile is not None and model is not None and is_owner_bound(profile):
                allowed, required_effort = _family_subscription_policy(profile, model)
                if not allowed or not is_grantable_subscription_model(profile, model, catalog):
                    raise PermissionError("This model is not assigned to your account.")
                requested_effort = str(selection.get("reasoning_effort") or "").strip().lower()
                if required_effort is not None and requested_effort not in {
                    "",
                    required_effort,
                }:
                    raise PermissionError(
                        "This subscription model uses the owner's fixed reasoning setting."
                    )
                resolved = dict(selection)
                if required_effort is None:
                    resolved.pop("reasoning_effort", None)
                else:
                    resolved["reasoning_effort"] = required_effort
                return resolved
            return selection
    raise PermissionError("This model is not assigned to your account.")
