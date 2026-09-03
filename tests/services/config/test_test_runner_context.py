"""Request identity propagation for background configuration probes."""

from __future__ import annotations

from threading import Event
from typing import Any

from deeptutor.multi_user.context import (
    get_current_user_or_none,
    reset_current_user,
    set_current_user,
)
from deeptutor.multi_user.models import CurrentUser, UserScope
from deeptutor.services.config.test_runner import ConfigTestRunner, TestRun


def test_start_preserves_current_user_in_worker_thread(tmp_path, monkeypatch) -> None:
    actor = CurrentUser(
        id="u_later_admin",
        username="later-admin@example.com",
        role="admin",
        scope=UserScope(
            kind="admin",
            user_id="u_later_admin",
            root=tmp_path / "admin",
        ),
    )
    catalog = {"services": {"llm": {"profiles": []}}}
    observed: list[tuple[CurrentUser | None, str, dict[str, Any]]] = []
    finished = Event()
    runner = ConfigTestRunner()

    def capture_identity(run: TestRun, resolved: dict[str, Any]) -> None:
        observed.append((get_current_user_or_none(), run.service, resolved))
        finished.set()

    monkeypatch.setattr(runner, "_run_sync", capture_identity)
    token = set_current_user(actor)
    try:
        run = runner.start("llm", catalog)
        assert finished.wait(timeout=2), "configuration probe worker did not start"
    finally:
        reset_current_user(token)

    assert runner.get(run.id) is run
    assert observed == [(actor, "llm", catalog)]
