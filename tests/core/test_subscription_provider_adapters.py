"""Native agent adapters for the adult operator's subscription providers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deeptutor.core.agentic import client as module
from deeptutor.multi_user import model_access
from deeptutor.services.llm.capabilities import supports_tools, supports_vision
from deeptutor.services.llm.provider_core.cursor_sdk_provider import CursorSDKProvider
from deeptutor.services.llm.provider_core.grok_subscription_provider import (
    GrokSubscriptionProvider,
)
from deeptutor.services.provider_registry import find_by_name


def _config(binding: str, model: str, api_key: str = "") -> module.LLMClientConfig:
    return module.LLMClientConfig(
        binding=binding,
        model=model,
        api_key=api_key,
        base_url=None,
        profile_id=f"{binding}-profile",
        model_id=f"{binding}-model",
    )


def test_cursor_subscription_agent_path_is_text_only() -> None:
    spec = find_by_name("cursor_subscription")
    assert spec is not None

    adapter = module._build_native_provider_adapter(
        _config("cursor_subscription", "cursor-grok-4.6-high", "cursor-key"),
        spec,
    )

    assert isinstance(adapter._provider, CursorSDKProvider)
    assert adapter._provider.api_key == "cursor-key"
    assert (
        module.can_use_native_tool_calling(
            binding="cursor_subscription", model="cursor-grok-4.6-high"
        )
        is False
    )
    assert supports_tools("cursor_subscription", "cursor-grok-4.6-high") is False
    assert supports_vision("cursor_subscription", "cursor-grok-4.6-high") is False


def test_grok_subscription_agent_path_is_text_only() -> None:
    spec = find_by_name("grok_subscription")
    assert spec is not None

    adapter = module._build_native_provider_adapter(
        _config("grok_subscription", "grok-4.6-high"),
        spec,
    )

    assert isinstance(adapter._provider, GrokSubscriptionProvider)
    assert (
        module.can_use_native_tool_calling(binding="grok_subscription", model="grok-4.6-high")
        is False
    )
    assert supports_tools("grok_subscription", "grok-4.6-high") is False
    assert supports_vision("grok_subscription", "grok-4.6-high") is False


def test_agent_client_rechecks_the_exact_subscription_model(monkeypatch) -> None:
    calls: list[tuple[str | None, str | None, str | None, str | None, str | None]] = []
    sentinel = object()
    config = _config("grok_subscription", "grok-4.6-high")
    monkeypatch.setattr(
        model_access,
        "require_deployment_owner_binding",
        lambda binding, user=None, model=None, profile_id=None, model_id=None, reasoning_effort=None: (
            calls.append((binding, model, profile_id, model_id, reasoning_effort))
        ),
    )
    monkeypatch.setattr(module, "load_system_settings", lambda: {"disable_ssl_verify": False})
    monkeypatch.setattr(
        module,
        "_build_openai_client",
        lambda _config, *, disable_ssl_verify: sentinel,
    )

    assert module.build_openai_client(config) is sentinel
    assert calls == [
        (
            "grok_subscription",
            "grok-4.6-high",
            "grok_subscription-profile",
            "grok_subscription-model",
            None,
        )
    ]


@pytest.mark.asyncio
async def test_cached_agent_adapter_rechecks_revoked_grant_at_stream_dispatch(
    monkeypatch,
) -> None:
    allowed = True
    dispatches = 0

    class FakeProvider:
        async def chat(self, **kwargs):
            nonlocal dispatches
            del kwargs
            dispatches += 1
            return SimpleNamespace(
                content="ok",
                tool_calls=[],
                finish_reason="stop",
                usage={},
                provider_specific_fields={},
            )

        async def chat_stream(self, **kwargs):
            nonlocal dispatches
            del kwargs
            dispatches += 1
            return SimpleNamespace(
                content="streamed",
                tool_calls=[],
                finish_reason="stop",
                usage={},
                provider_specific_fields={},
            )

    config = _config("grok_subscription", "grok-4.6-high")

    def require(
        binding,
        user=None,
        model=None,
        profile_id=None,
        model_id=None,
        reasoning_effort=None,
    ):
        del user
        assert (binding, model, profile_id, model_id) == (
            "grok_subscription",
            "grok-4.6-high",
            "grok_subscription-profile",
            "grok_subscription-model",
        )
        if not allowed:
            raise PermissionError("revoked")

    adapter = module._ProviderOpenAIAdapter(
        FakeProvider(),
        binding=config.binding,
        model=config.model,
        profile_id=config.profile_id,
        model_id=config.model_id,
    )
    monkeypatch.setattr(model_access, "require_deployment_owner_binding", require)
    monkeypatch.setattr(module, "load_system_settings", lambda: {"disable_ssl_verify": False})
    monkeypatch.setattr(
        module,
        "_build_openai_client",
        lambda _config, *, disable_ssl_verify: adapter,
    )
    module.reset_agentic_client_pool()

    client = module.build_openai_client(config)
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "first"}],
        model="grok-4.6-high",
    )
    assert response.choices[0].message.content == "ok"
    stream = await client.chat.completions.create(
        messages=[{"role": "user", "content": "second"}],
        model="grok-4.6-high",
        stream=True,
    )
    allowed = False

    with pytest.raises(PermissionError, match="revoked"):
        async for _chunk in stream:
            pass
    assert dispatches == 1
    await module.close_agentic_client_pool()
