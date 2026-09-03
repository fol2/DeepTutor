"""Native agent adapters for the adult operator's subscription providers."""

from __future__ import annotations

from deeptutor.core.agentic import client as module
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
