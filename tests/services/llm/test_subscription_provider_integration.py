"""Registry, config and factory wiring for subscription-backed providers."""

from __future__ import annotations

from deeptutor.services.config.provider_runtime import ResolvedLLMConfig
from deeptutor.services.llm import config as config_module
from deeptutor.services.llm import provider_factory
from deeptutor.services.llm.config import LLMConfig
from deeptutor.services.llm.provider_core.cursor_sdk_provider import CursorSDKProvider
from deeptutor.services.llm.provider_core.grok_subscription_provider import (
    GrokSubscriptionProvider,
)
from deeptutor.services.provider_registry import find_by_name


def _config(binding: str, model: str, api_key: str = "") -> LLMConfig:
    return LLMConfig(
        model=model,
        api_key=api_key,
        binding=binding,
        provider_name=binding,
        provider_mode=find_by_name(binding).mode,  # type: ignore[union-attr]
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
