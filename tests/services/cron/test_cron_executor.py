"""Focused identity restoration tests for scheduled chat execution."""

import pytest

from deeptutor.core.stream import StreamEvent, StreamEventType
from deeptutor.multi_user.context import get_current_user
from deeptutor.multi_user.paths import admin_scope, scope_for_user
from deeptutor.services.cron import executor
from deeptutor.services.cron.executor import _current_user_for_chat_owner
from deeptutor.services.cron.service import CronJob, CronOwner, CronSchedule


@pytest.mark.asyncio
async def test_admin_owner_identity_is_preserved_for_scheduled_chat(monkeypatch) -> None:
    from deeptutor.multi_user import identity

    monkeypatch.setattr(
        identity,
        "get_user_by_id",
        lambda _user_id: ("owner@example.com", {"id": "admin-account-7", "role": "admin"}),
    )
    owner = CronOwner(kind="chat", user_id="admin-account-7", is_admin=True)

    user = await _current_user_for_chat_owner(owner)

    assert user is not None
    assert user.id == "admin-account-7"
    assert user.username == "owner@example.com"
    assert user.role == "admin"
    assert user.scope.kind == "admin"
    assert user.scope.user_id == "admin-account-7"
    assert user.scope.root == admin_scope().root


@pytest.mark.asyncio
async def test_ordinary_owner_identity_and_scope_are_preserved_for_scheduled_chat(
    monkeypatch,
) -> None:
    from deeptutor.multi_user import identity

    monkeypatch.setattr(
        identity,
        "get_user_by_id",
        lambda _user_id: (
            "learner@example.com",
            {"id": "learner-account-4", "role": "user"},
        ),
    )
    owner = CronOwner(kind="chat", user_id="learner-account-4", is_admin=False)

    user = await _current_user_for_chat_owner(owner)

    assert user is not None
    expected_scope = scope_for_user("learner-account-4", is_admin=False)
    assert user.id == "learner-account-4"
    assert user.username == "learner@example.com"
    assert user.role == "user"
    assert user.scope == expected_scope


@pytest.mark.asyncio
async def test_scheduled_chat_runs_inside_actual_admin_owner_context(monkeypatch) -> None:
    observed_users = []

    from deeptutor.multi_user import identity

    monkeypatch.setattr(
        identity,
        "get_user_by_id",
        lambda _user_id: ("owner@example.com", {"id": "admin-account-7", "role": "admin"}),
    )

    class FakeStore:
        async def get_session(self, _session_id):
            return object()

        async def get_messages_for_context(self, _session_id):
            return []

        async def add_message(self, **_kwargs):
            return None

    class FakeOrchestrator:
        async def handle(self, _context):
            observed_users.append(get_current_user())
            yield StreamEvent(
                type=StreamEventType.RESULT,
                source="chat",
                metadata={"response": "Done"},
            )

    async def no_notification(*_args, **_kwargs):
        return None

    import deeptutor.runtime.orchestrator as orchestrator_module
    import deeptutor.services.session as session_module

    monkeypatch.setattr(orchestrator_module, "ChatOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(session_module, "get_sqlite_session_store", lambda: FakeStore())
    monkeypatch.setattr(executor, "_maybe_send_desktop_notification", no_notification)

    job = CronJob(
        id="scheduled-owner-check",
        name="Owner check",
        message="Run as me",
        schedule=CronSchedule(kind="every", every_seconds=3600),
        owner=CronOwner(
            kind="chat",
            user_id="admin-account-7",
            is_admin=True,
            session_id="session-1",
        ),
    )

    assert await executor.execute_job(job) == ("ok", None)
    assert len(observed_users) == 1
    assert observed_users[0].id == "admin-account-7"
    assert observed_users[0].scope.user_id == "admin-account-7"


@pytest.mark.asyncio
async def test_scheduled_admin_demotion_uses_current_ordinary_user_role(monkeypatch) -> None:
    from deeptutor.multi_user import identity

    monkeypatch.setattr(
        identity,
        "get_user_by_id",
        lambda _user_id: ("former@example.com", {"id": "former-owner", "role": "user"}),
    )
    persisted_owner = CronOwner(kind="chat", user_id="former-owner", is_admin=True)

    user = await _current_user_for_chat_owner(persisted_owner)

    assert user is not None
    assert user.role == "user"
    assert user.scope.kind == "user"


@pytest.mark.asyncio
async def test_deleted_scheduled_owner_is_skipped_before_turn_execution(monkeypatch) -> None:
    from deeptutor.multi_user import identity
    import deeptutor.runtime.orchestrator as orchestrator_module

    monkeypatch.setattr(identity, "get_user_by_id", lambda _user_id: None)

    class UnexpectedOrchestrator:
        def __init__(self):
            raise AssertionError("orchestrator must not start for a deleted owner")

    monkeypatch.setattr(orchestrator_module, "ChatOrchestrator", UnexpectedOrchestrator)
    job = CronJob(
        id="deleted-owner",
        name="Deleted owner",
        message="Do not run",
        schedule=CronSchedule(kind="every", every_seconds=3600),
        owner=CronOwner(
            kind="chat",
            user_id="deleted-owner-id",
            is_admin=True,
            session_id="session-1",
        ),
    )

    assert await executor.execute_job(job) == (
        "skipped",
        "owner account is unavailable",
    )


@pytest.mark.asyncio
async def test_scheduled_learner_resolves_current_grant_before_turn(monkeypatch) -> None:
    """Cron stores no model: the owner scope resolves a fresh exact grant."""
    observed_selection = []
    reset_tokens = []

    from deeptutor.multi_user import identity, model_access
    import deeptutor.runtime.orchestrator as orchestrator_module
    from deeptutor.services.model_selection import runtime as selection_runtime
    import deeptutor.services.session as session_module

    monkeypatch.setattr(
        identity,
        "get_user_by_id",
        lambda _user_id: ("learner@example.com", {"id": "learner-7", "role": "user"}),
    )
    selection = {"profile_id": "cursor", "model_id": "grok-high"}
    monkeypatch.setattr(
        model_access,
        "default_allowed_llm_selection",
        lambda user_id: selection if user_id == "learner-7" else None,
    )
    token = object()

    def activate(candidate):
        observed_selection.append(candidate)
        return object(), token

    monkeypatch.setattr(selection_runtime, "activate_llm_selection", activate)
    monkeypatch.setattr(selection_runtime, "reset_llm_selection", reset_tokens.append)

    class FakeStore:
        async def get_session(self, _session_id):
            return object()

        async def get_messages_for_context(self, _session_id):
            return []

        async def add_message(self, **_kwargs):
            return None

    class FakeOrchestrator:
        async def handle(self, _context):
            yield StreamEvent(
                type=StreamEventType.RESULT,
                source="chat",
                metadata={"response": "Done"},
            )

    async def no_notification(*_args, **_kwargs):
        return None

    monkeypatch.setattr(orchestrator_module, "ChatOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(session_module, "get_sqlite_session_store", lambda: FakeStore())
    monkeypatch.setattr(executor, "_maybe_send_desktop_notification", no_notification)
    job = CronJob(
        id="learner-grant",
        name="Learner grant",
        message="Use my assigned model",
        schedule=CronSchedule(kind="every", every_seconds=3600),
        owner=CronOwner(kind="chat", user_id="learner-7", session_id="session-1"),
    )

    assert await executor.execute_job(job) == ("ok", None)
    assert observed_selection == [selection]
    assert reset_tokens == [token]


@pytest.mark.asyncio
async def test_scheduled_learner_with_revoked_grant_never_starts_turn(monkeypatch) -> None:
    from deeptutor.multi_user import identity, model_access
    import deeptutor.runtime.orchestrator as orchestrator_module

    monkeypatch.setattr(
        identity,
        "get_user_by_id",
        lambda _user_id: ("learner@example.com", {"id": "learner-7", "role": "user"}),
    )
    monkeypatch.setattr(model_access, "default_allowed_llm_selection", lambda _user_id: None)

    class UnexpectedOrchestrator:
        def __init__(self):
            raise AssertionError("cron must not use the global model after grant revocation")

    monkeypatch.setattr(orchestrator_module, "ChatOrchestrator", UnexpectedOrchestrator)
    job = CronJob(
        id="learner-revoked",
        name="Learner revoked",
        message="Do not run",
        schedule=CronSchedule(kind="every", every_seconds=3600),
        owner=CronOwner(kind="chat", user_id="learner-7", session_id="session-1"),
    )

    assert await executor.execute_job(job) == (
        "skipped",
        "no LLM model is assigned to the account",
    )
