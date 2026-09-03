"""A local JWT reflects current account state, not its issuance-time role."""

from __future__ import annotations


def test_local_token_reloads_role_and_is_revoked_on_disable_or_delete(monkeypatch) -> None:
    from deeptutor.services import auth

    records = {
        "owner@example.com": {
            "id": "u_owner",
            "hash": "unused",
            "role": "admin",
            "disabled": False,
        }
    }
    monkeypatch.setattr(auth, "AUTH_SECRET", "test-secret-with-enough-entropy")
    monkeypatch.setattr(auth, "POCKETBASE_ENABLED", False)
    monkeypatch.setattr(auth, "_load_users", lambda: records)
    token = auth.create_token("owner@example.com", role="admin", user_id="u_owner")

    original = auth.decode_token(token)
    assert original is not None and original.role == "admin"

    records["owner@example.com"]["role"] = "user"
    demoted = auth.decode_token(token)
    assert demoted is not None and demoted.role == "user"

    records["owner@example.com"]["disabled"] = True
    assert auth.decode_token(token) is None

    records.clear()
    assert auth.decode_token(token) is None


def test_local_token_is_revoked_when_username_is_recreated_with_new_id(monkeypatch) -> None:
    from deeptutor.services import auth

    records = {
        "owner@example.com": {
            "id": "u_owner",
            "hash": "unused",
            "role": "admin",
            "disabled": False,
        }
    }
    monkeypatch.setattr(auth, "AUTH_SECRET", "test-secret-with-enough-entropy")
    monkeypatch.setattr(auth, "POCKETBASE_ENABLED", False)
    monkeypatch.setattr(auth, "_load_users", lambda: records)
    token = auth.create_token("owner@example.com", role="admin", user_id="u_owner")

    records["owner@example.com"]["id"] = "u_recreated"

    assert auth.decode_token(token) is None
