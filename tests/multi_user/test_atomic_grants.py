"""Concurrency regressions for protected subscription grant updates."""

from __future__ import annotations

from threading import Event, Thread

from deeptutor.multi_user import grants, model_access
from deeptutor.multi_user.models import CurrentUser, UserScope


def _admin(tmp_path, user_id: str) -> CurrentUser:
    return CurrentUser(
        id=user_id,
        username=f"{user_id}@example.com",
        role="admin",
        scope=UserScope(kind="admin", user_id=user_id, root=tmp_path / "admin"),
    )


def test_concurrent_later_admin_edit_cannot_resurrect_owner_revocation(
    tmp_path, monkeypatch
) -> None:
    user_id = "u_child"
    profile_id = "llm-profile-cursor-subscription"
    model_id = "llm-model-cursor-grok-high"
    catalog = {
        model_access.DEPLOYMENT_OWNER_ID_FIELD: "u_owner",
        "services": {
            "llm": {
                "profiles": [
                    {
                        "id": profile_id,
                        "binding": "cursor_subscription",
                        "owner_bound": True,
                        "models": [
                            {
                                "id": model_id,
                                "name": "Grok 4.6 High",
                                "model": "cursor-grok-4.6-high",
                            }
                        ],
                    }
                ]
            }
        },
    }
    monkeypatch.setattr(grants, "GRANTS_DIR", tmp_path / "grants")
    monkeypatch.setattr(
        grants,
        "get_user_by_id",
        lambda requested: (
            ("child@example.com", {"id": requested, "role": "user"})
            if requested == user_id
            else None
        ),
    )
    grants.save_grant(
        user_id,
        {
            "models": {
                "llm": [
                    {
                        "profile_id": profile_id,
                        "model_ids": [model_id],
                        model_access.SUBSCRIPTION_GRANT_ISSUER_FIELD: "u_owner",
                    }
                ]
            }
        },
    )

    later_admin_loaded = Event()
    allow_later_admin_write = Event()
    owner_write_started = Event()
    failures: list[BaseException] = []

    def later_admin_update() -> None:
        try:

            def prepare(existing: dict) -> dict:
                later_admin_loaded.set()
                assert allow_later_admin_write.wait(timeout=5)
                return model_access.prepare_assignable_model_grants(
                    {"models": {"llm": []}, "knowledge_bases": [{"resource_id": "kb:new"}]},
                    existing=existing,
                    user=_admin(tmp_path, "u_later_admin"),
                    catalog=catalog,
                )

            grants.update_grant(user_id, prepare)
        except BaseException as exc:  # pragma: no cover - reported in the main thread
            failures.append(exc)

    def owner_revoke() -> None:
        try:
            owner_write_started.set()
            grants.save_grant(user_id, {"models": {"llm": []}})
        except BaseException as exc:  # pragma: no cover - reported in the main thread
            failures.append(exc)

    later_thread = Thread(target=later_admin_update)
    later_thread.start()
    assert later_admin_loaded.wait(timeout=5)

    owner_thread = Thread(target=owner_revoke)
    owner_thread.start()
    assert owner_write_started.wait(timeout=5)
    assert owner_thread.is_alive(), "owner revocation should wait for the in-flight atomic update"

    allow_later_admin_write.set()
    later_thread.join(timeout=5)
    owner_thread.join(timeout=5)

    assert not later_thread.is_alive()
    assert not owner_thread.is_alive()
    assert failures == []
    final = grants.load_grant(user_id)
    assert final["models"]["llm"] == []
    assert final["knowledge_bases"] == []
