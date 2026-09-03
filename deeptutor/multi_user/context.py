"""Request-local current user context."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
import logging
from typing import Any

from .models import LOCAL_ADMIN_ID, CurrentUser, UserScope
from .paths import local_admin_user, scope_for_user

logger = logging.getLogger(__name__)

_current_user: ContextVar[CurrentUser | None] = ContextVar("deeptutor_current_user", default=None)


def set_current_user(user: CurrentUser) -> Token[CurrentUser | None]:
    return _current_user.set(user)


def reset_current_user(token: Token[CurrentUser | None]) -> None:
    _current_user.reset(token)


def get_current_user() -> CurrentUser:
    return _current_user.get() or local_admin_user()


def get_current_user_or_none() -> CurrentUser | None:
    return _current_user.get()


async def resolve_current_user_by_id(user_id: str) -> CurrentUser | None:
    """Resolve an account's current role/status for background work.

    Persisted jobs and partners carry an id, never a trusted role snapshot.
    Re-read the configured identity backend before every run so a disabled,
    deleted or demoted account cannot retain ambient process privileges.
    """
    from deeptutor.services import auth as auth_service

    from .identity import get_user_by_id

    owner_id = user_id or LOCAL_ADMIN_ID

    def resolved_scope(*, is_admin: bool) -> UserScope:
        scope = scope_for_user(owner_id, is_admin=is_admin)
        if is_admin and scope.user_id != owner_id:
            return UserScope(kind=scope.kind, user_id=owner_id, root=scope.root)
        return scope

    if owner_id == LOCAL_ADMIN_ID:
        return local_admin_user() if not auth_service.AUTH_ENABLED else None

    if owner_id == "env-admin":
        if (
            auth_service.AUTH_ENABLED
            and not auth_service.POCKETBASE_ENABLED
            and auth_service.AUTH_USERNAME
            and auth_service.AUTH_PASSWORD_HASH
        ):
            return CurrentUser(
                id=owner_id,
                username=auth_service.AUTH_USERNAME,
                role="admin",
                scope=resolved_scope(is_admin=True),
            )
        return None

    if auth_service.POCKETBASE_ENABLED:
        try:
            from deeptutor.services.pocketbase_client import get_pb_client

            record = await asyncio.to_thread(
                get_pb_client().collection("users").get_one,
                owner_id,
            )
        except Exception:
            logger.warning("Account %s could not be resolved from PocketBase", owner_id)
            return None
        if bool(getattr(record, "disabled", False)):
            return None
        username = str(
            getattr(record, "email", None)
            or getattr(record, "name", None)
            or getattr(record, "username", None)
            or owner_id
        )
        role = "admin" if str(getattr(record, "role", "user") or "user") == "admin" else "user"
        return CurrentUser(
            id=owner_id,
            username=username,
            role=role,
            scope=resolved_scope(is_admin=role == "admin"),
        )

    found = get_user_by_id(owner_id)
    if found is None:
        return None
    username, record = found
    if record.get("disabled"):
        return None
    role = "admin" if str(record.get("role") or "user") == "admin" else "user"
    return CurrentUser(
        id=owner_id,
        username=username,
        role=role,
        scope=resolved_scope(is_admin=role == "admin"),
    )


def user_from_token_payload(payload: Any | None) -> CurrentUser:
    if payload is None:
        return local_admin_user()
    user_id = str(getattr(payload, "user_id", "") or "")
    username = str(getattr(payload, "username", "") or "local")
    role = str(getattr(payload, "role", "user") or "user")
    if role not in {"admin", "user"}:
        role = "user"
    if not user_id:
        user_id = "local-admin" if role == "admin" and username == "local" else username
    return CurrentUser(
        id=user_id,
        username=username,
        role=role,  # type: ignore[arg-type]
        scope=scope_for_user(user_id, is_admin=role == "admin"),
    )
