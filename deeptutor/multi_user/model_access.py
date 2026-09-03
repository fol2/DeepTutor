"""Server-side model grant resolution and redacted model views.

Grants carry LLM assignments only (grant v2): embedding and search always
resolve from the deployment's active profiles, so per-user grants for them
were never enforced and are not stored.

Only administrator-assigned models reach an ordinary user, and
:func:`redacted_model_access` is the one place those grants are resolved.
Consumer-subscription connectors are deployment-owner-only and are never
grantable. Everything downstream — the option list, the capability gate, and
selection validation — reads that one function, so the three can never
disagree about what a user may use.
"""

from __future__ import annotations

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
DEPLOYMENT_OWNER_ID_FIELD = "deployment_owner_user_id"
_OWNER_BOUND_COMPACT_BINDINGS = frozenset(
    binding.replace("_", "") for binding in OWNER_BOUND_BINDINGS
)
_DEPLOYMENT_OWNER_COMPACT_BINDINGS = frozenset(
    binding.replace("_", "") for binding in DEPLOYMENT_OWNER_BINDINGS
)


def _binding_keys(value: Any) -> tuple[str | None, str]:
    raw_binding = str(value or "").strip()
    canonical = canonical_provider_name(raw_binding)
    compact = "".join(character for character in raw_binding.casefold() if character.isalnum())
    return canonical, compact


def is_owner_bound(profile: dict[str, Any]) -> bool:
    """Whether a profile is tied to the identity of the operator who set it up.

    OAuth providers such as Codex authenticate one individual's plan rather than
    a billable team key, so those profiles are never lent to other accounts
    through grants.
    """
    binding, compact_binding = _binding_keys(profile.get("binding") or profile.get("provider"))
    if binding in OWNER_BOUND_BINDINGS or compact_binding in _OWNER_BOUND_COMPACT_BINDINGS:
        return True
    return bool(profile.get("owner_bound"))


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
) -> None:
    """Reject deployment-global subscription state for every non-owner."""
    if not is_deployment_owner_binding(binding):
        return
    if not is_deployment_owner(user):
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
        if profile is not None and is_owner_bound(profile):
            # A grant may predate the profile becoming owner-bound. Drop it here,
            # the one place every caller resolves grants through, so the option
            # list, the capability gate, and selection validation all agree.
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
            return selection
    raise PermissionError("This model is not assigned to your account.")
