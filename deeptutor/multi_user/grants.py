"""Logical resource grants for non-admin users."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from threading import Lock, RLock
from typing import Any, Callable

from .identity import get_user_by_id
from .paths import SYSTEM_ROOT, ensure_system_dirs

GRANTS_DIR = SYSTEM_ROOT / "grants"
_GRANT_LOCKS: dict[str, RLock] = {}
_GRANT_LOCKS_GUARD = Lock()


def _grant_lock(user_id: str) -> RLock:
    """Return the process-local lock serialising one user's grant updates."""
    with _GRANT_LOCKS_GUARD:
        return _GRANT_LOCKS.setdefault(user_id, RLock())


def empty_grant(user_id: str) -> dict[str, Any]:
    return {
        "version": 2,
        "user_id": user_id,
        "models": {"llm": []},
        "knowledge_bases": [],
        "skills": [],
        # Partners an admin has lent this user. People build their own partners
        # now, so a grant is only about someone *else's*: it lets the user talk
        # to the named partners — never configure them — and their side of each
        # conversation stays private to their account. Same shape as ``skills``
        # (``[{"partner_id": ...}]``).
        "partners": [],
        # Tool whitelists share the partner-config semantics for built-ins:
        # ``enabled_tools=None`` means "default" (every tool in the pool),
        # ``[]`` means none, a list is an explicit whitelist. MCP tools can
        # proxy host-side capabilities, so non-admin runtime access treats
        # ``mcp_tools=None`` as deny-by-default until an admin grants explicit
        # names. ``cli_apps`` is the same posture for installed CLI apps, keyed
        # by app id: each one is third-party code executing in the sandbox, so
        # an absent grant is no access rather than all of them.
        # ``exec_enabled`` is a tri-state override on top of the
        # deployment exec policy: ``None`` follows the policy, ``False`` always
        # denies, ``True`` is only honored where the sandbox can actually
        # isolate users (SYSTEM isolation).
        "enabled_tools": None,
        "mcp_tools": None,
        "cli_apps": None,
        "exec_enabled": None,
    }


def _normalize_tool_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    return [str(item).strip() for item in value if str(item).strip()]


def grant_path(user_id: str) -> Path:
    ensure_system_dirs()
    return GRANTS_DIR / f"{user_id}.json"


def normalize_grant(user_id: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce any stored/submitted grant payload into the v2 shape.

    v1 grants normalize losslessly for everything that was ever enforced:
    ``models.embedding`` / ``models.search`` / ``spaces`` had no runtime
    consumers and are dropped; absent v2 fields default to unrestricted.
    """
    base = empty_grant(user_id)
    if not isinstance(payload, dict):
        return base
    base["user_id"] = user_id
    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    items = models.get("llm") if isinstance(models, dict) else []
    if not isinstance(items, list):
        items = []
    base["models"]["llm"] = [dict(item) for item in items if isinstance(item, dict)]
    for key in ("knowledge_bases", "skills", "partners"):
        # Read once, then narrow. Two separate ``.get`` calls cannot be narrowed
        # together — nothing promises they return the same object — so the
        # inline-conditional form left this iterating a possible ``None``.
        raw = payload.get(key)
        values = raw if isinstance(raw, list) else []
        base[key] = [dict(item) for item in values if isinstance(item, dict)]
    for key in ("enabled_tools", "mcp_tools", "cli_apps"):
        base[key] = _normalize_tool_list(payload.get(key))
    exec_enabled = payload.get("exec_enabled")
    base["exec_enabled"] = bool(exec_enabled) if isinstance(exec_enabled, bool) else None
    return base


def _load_grant_unlocked(user_id: str) -> dict[str, Any]:
    path = grant_path(user_id)
    if not path.exists():
        return empty_grant(user_id)
    try:
        return normalize_grant(user_id, json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return empty_grant(user_id)


def load_grant(user_id: str) -> dict[str, Any]:
    with _grant_lock(user_id):
        return _load_grant_unlocked(user_id)


def _validated_grant(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    user_record = get_user_by_id(user_id)
    if user_record is None:
        raise ValueError(f"Unknown user id: {user_id}")
    _username, record = user_record
    if str(record.get("role") or "user") == "admin":
        raise ValueError("Admin users use the main workspace and cannot receive assignments.")
    grant = normalize_grant(user_id, payload)
    validate_grant(grant)
    return grant


def _atomic_write_grant(user_id: str, grant: dict[str, Any]) -> None:
    path = grant_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(grant, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_grant(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and atomically replace one stored grant."""
    with _grant_lock(user_id):
        grant = _validated_grant(user_id, payload)
        _atomic_write_grant(user_id, grant)
        return grant


def update_grant(
    user_id: str,
    update: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Apply a read-modify-write grant update under one per-user lock.

    The callback receives a fresh, detached snapshot. Validation and the final
    atomic file replacement remain inside the same critical section, so a
    stale administrator edit cannot overwrite a concurrent revocation.
    """
    with _grant_lock(user_id):
        existing = deepcopy(_load_grant_unlocked(user_id))
        grant = _validated_grant(user_id, update(existing))
        _atomic_write_grant(user_id, grant)
        return grant


def validate_grant(grant: dict[str, Any]) -> None:
    """Reject accidental secret/path material in grants.

    Grants carry logical ids only. Runtime resolution happens server-side.
    """
    forbidden = {"api_key", "secret", "password", "token", "path", "base_url"}

    def walk(value: Any, trail: str = "grant") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = str(key).lower()
                if lowered in forbidden or lowered.endswith("_key"):
                    raise ValueError(f"Grants must not contain secret/path field: {trail}.{key}")
                walk(child, f"{trail}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{trail}[{index}]")

    walk(grant)


def public_grant(user_id: str) -> dict[str, Any]:
    return deepcopy(load_grant(user_id))
