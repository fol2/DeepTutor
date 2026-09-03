"""Focused identity restoration tests for scheduled chat execution."""

import pytest

from deeptutor.core.stream import StreamEvent, StreamEventType
from deeptutor.multi_user.context import get_current_user
from deeptutor.multi_user.paths import admin_scope, scope_for_user
from deeptutor.services.cron import executor
from deeptutor.services.cron.executor import _current_user_for_chat_owner
from deeptutor.services.cron.service import CronJob, CronOwner, CronSchedule


def test_admin_owner_identity_is_preserved_for_scheduled_chat() -> None:
    owner = CronOwner(kind="chat", user_id="admin-account-7", is_admin=True)

    user = _current_user_for_chat_owner(owner)

    assert user.id == "admin-account-7"
    assert user.username == "admin-account-7"
    assert user.role == "admin"
    assert user.scope.kind == "admin"
    assert user.scope.user_id == "admin-account-7"
    assert user.scope.root == admin_scope().root


def test_ordinary_owner_identity_and_scope_are_preserved_for_scheduled_chat() -> None:
    owner = CronOwner(kind="chat", user_id="learner-account-4", is_admin=False)

    user = _current_user_for_chat_owner(owner)

    expected_scope = scope_for_user("learner-account-4", is_admin=False)
    assert user.id == "learner-account-4"
    assert user.username == "learner-account-4"
    assert user.role == "user"
    assert user.scope == expected_scope


@pytest.mark.asyncio
async def test_scheduled_chat_runs_inside_actual_admin_owner_context(monkeypatch) -> None:
    observed_users = []

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
