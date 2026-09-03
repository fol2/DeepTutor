"""Registry, config and factory wiring for subscription-backed providers."""

from __future__ import annotations

import pytest

from deeptutor.multi_user import model_access
from deeptutor.services.config.provider_runtime import ResolvedLLMConfig
from deeptutor.services.llm import config as config_module
from deeptutor.services.llm import provider_factory
from deeptutor.services.llm.config import LLMConfig
from deeptutor.services.llm.provider_core.base import LLMResponse
from deeptutor.services.llm.provider_core.cursor_sdk_provider import CursorSDKProvider
from deeptutor.services.llm.provider_core.grok_subscription_provider import (
    GrokSubscriptionProvider,
)
from deeptutor.services.llm.provider_core.openai_codex_provider import OpenAICodexProvider
from deeptutor.services.provider_registry import find_by_name


def _config(binding: str, model: str, api_key: str = "") -> LLMConfig:
    return LLMConfig(
        model=model,
        api_key=api_key,
        binding=binding,
        provider_name=binding,
        provider_mode=find_by_name(binding).mode,  # type: ignore[union-attr]
        profile_id=f"{binding}-profile",
        model_id=f"{binding}-model",
        reasoning_effort="max" if binding == "openai_codex" else None,
    )


def test_runtime_factory_builds_cursor_and_grok_native_providers() -> None:
    cursor = provider_factory._build_runtime_provider(
        _config("cursor_subscription", "cursor-grok-4.6-high", "cursor-key")
    )
    grok = provider_factory._build_runtime_provider(_config("grok_subscription", "grok-4.6-high"))

    assert isinstance(cursor, CursorSDKProvider)
    assert cursor.api_key == "cursor-key"
    assert isinstance(grok, GrokSubscriptionProvider)


def test_cursor_native_provider_does_not_require_a_fake_http_endpoint(monkeypatch) -> None:
    resolved = ResolvedLLMConfig(
        model="cursor-grok-4.6-high",
        provider_name="cursor_subscription",
        provider_mode="direct",
        binding="cursor_subscription",
        api_key="cursor-key",
        base_url=None,
        effective_url=None,
    )
    monkeypatch.setattr(config_module, "resolve_llm_runtime_config", lambda: resolved)

    config = config_module._get_llm_config_from_resolver()

    assert config.provider_name == "cursor_subscription"
    assert config.effective_url is None


def test_runtime_factory_rechecks_the_exact_subscription_model(monkeypatch) -> None:
    calls: list[tuple[str | None, str | None, str | None, str | None, str | None]] = []
    sentinel = object()
    config = _config("cursor_subscription", "cursor-grok-4.6-high", "cursor-key")
    monkeypatch.setattr(
        model_access,
        "require_deployment_owner_binding",
        lambda binding, user=None, model=None, profile_id=None, model_id=None, reasoning_effort=None: (
            calls.append((binding, model, profile_id, model_id, reasoning_effort))
        ),
    )
    monkeypatch.setattr(provider_factory, "_build_runtime_provider", lambda _config: sentinel)

    assert provider_factory.get_runtime_provider(config) is sentinel
    assert calls == [
        (
            "cursor_subscription",
            "cursor-grok-4.6-high",
            "cursor_subscription-profile",
            "cursor_subscription-model",
            None,
        )
    ]


@pytest.mark.asyncio
async def test_cached_cursor_provider_rechecks_revoked_grant_before_sdk_dispatch(
    monkeypatch,
) -> None:
    allowed = True
    dispatches = 0

    def require(
        binding,
        user=None,
        model=None,
        profile_id=None,
        model_id=None,
        reasoning_effort=None,
    ):
        del user
        assert (binding, model, profile_id, model_id, reasoning_effort) == (
            "cursor_subscription",
            "cursor-grok-4.6-high",
            "cursor_subscription-profile",
            "cursor_subscription-model",
            None,
        )
        if not allowed:
            raise PermissionError("revoked")

    async def call_cursor(*args, **kwargs):
        nonlocal dispatches
        del args, kwargs
        dispatches += 1
        return LLMResponse(content="ok", finish_reason="stop")

    monkeypatch.setattr(model_access, "require_deployment_owner_binding", require)
    monkeypatch.setattr(CursorSDKProvider, "_call_cursor", call_cursor)
    provider_factory.reset_runtime_provider_pool()
    config = _config("cursor_subscription", "cursor-grok-4.6-high", "cursor-key")

    provider = provider_factory.get_runtime_provider(config)
    assert await provider.chat([{"role": "user", "content": "first"}]) == LLMResponse(
        content="ok", finish_reason="stop"
    )
    assert provider_factory.get_runtime_provider(config) is provider
    allowed = False

    try:
        await provider.chat([{"role": "user", "content": "second"}])
    except PermissionError as exc:
        assert str(exc) == "revoked"
    else:  # pragma: no cover - proves fail-closed revocation
        raise AssertionError("revoked grant reached Cursor SDK dispatch")
    assert dispatches == 1
    await provider_factory.close_runtime_provider_pool()


@pytest.mark.asyncio
async def test_grok_provider_rechecks_revoked_grant_before_cli_dispatch(monkeypatch) -> None:
    allowed = True
    dispatches = 0

    def require(
        binding,
        user=None,
        model=None,
        profile_id=None,
        model_id=None,
        reasoning_effort=None,
    ):
        del user
        assert (binding, model, profile_id, model_id, reasoning_effort) == (
            "grok_subscription",
            "grok-4.6-high",
            "grok_subscription-profile",
            "grok_subscription-model",
            None,
        )
        if not allowed:
            raise PermissionError("revoked")

    async def run_cli(*args, **kwargs):
        nonlocal dispatches
        del args, kwargs
        dispatches += 1
        return "ok", False

    monkeypatch.setattr(model_access, "require_deployment_owner_binding", require)
    monkeypatch.setattr(GrokSubscriptionProvider, "_run_cli", run_cli)
    provider = provider_factory._build_runtime_provider(
        _config("grok_subscription", "grok-4.6-high")
    )

    assert (await provider.chat([{"role": "user", "content": "first"}])).content == "ok"
    allowed = False
    with pytest.raises(PermissionError, match="revoked"):
        await provider.chat([{"role": "user", "content": "second"}])
    assert dispatches == 1


@pytest.mark.asyncio
async def test_codex_provider_rechecks_revoked_grant_before_oauth_dispatch(monkeypatch) -> None:
    allowed = True
    dispatches = 0

    def require(
        binding,
        user=None,
        model=None,
        profile_id=None,
        model_id=None,
        reasoning_effort=None,
    ):
        del user
        assert (binding, model, profile_id, model_id, reasoning_effort) == (
            "openai_codex",
            "gpt-5.6-luna",
            "openai_codex-profile",
            "openai_codex-model",
            "max",
        )
        if not allowed:
            raise PermissionError("revoked")

    async def call_codex(*args, **kwargs):
        nonlocal dispatches
        del args, kwargs
        dispatches += 1
        return LLMResponse(content="ok", finish_reason="stop")

    monkeypatch.setattr(model_access, "require_deployment_owner_binding", require)
    monkeypatch.setattr(OpenAICodexProvider, "_call_codex", call_codex)
    provider = provider_factory._build_runtime_provider(_config("openai_codex", "gpt-5.6-luna"))

    assert (
        await provider.chat(
            [{"role": "user", "content": "first"}],
            reasoning_effort="max",
        )
    ).content == "ok"
    allowed = False
    with pytest.raises(PermissionError, match="revoked"):
        await provider.chat(
            [{"role": "user", "content": "second"}],
            reasoning_effort="max",
        )
    assert dispatches == 1
