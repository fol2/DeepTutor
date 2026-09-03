"""Runtime helpers for request-scoped model selection."""

from __future__ import annotations

from contextvars import Token
from dataclasses import dataclass
from typing import Any

from deeptutor.multi_user.models import CurrentUser
from deeptutor.services.config.provider_runtime import ResolvedLLMConfig, resolve_llm_runtime_config
from deeptutor.services.llm import config as llm_config_module
from deeptutor.services.llm.config import LLMConfig


def llm_config_from_resolved(resolved: ResolvedLLMConfig) -> LLMConfig:
    """Convert provider-runtime output into the LLM service config shape."""
    return LLMConfig(
        model=resolved.model,
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        effective_url=resolved.effective_url,
        binding=resolved.binding,
        provider_name=resolved.provider_name,
        provider_mode=resolved.provider_mode,
        profile_id=resolved.profile_id,
        model_id=resolved.model_id,
        api_version=resolved.api_version,
        extra_headers=resolved.extra_headers,
        reasoning_effort=resolved.reasoning_effort,
        context_window=resolved.context_window,
    )


def resolve_llm_config_for_selection(selection: Any) -> LLMConfig:
    """Resolve an access-checked LLM config for the current user.

    A selection can outlive its request in a queued memory run or scheduled
    chat.  Re-resolve the current user's logical grant immediately before the
    config is scoped: a revoked or deleted assignment must not silently fall
    through to the deployment-wide active config.
    """
    from deeptutor.multi_user.context import get_current_user
    from deeptutor.multi_user.model_access import (
        apply_allowed_llm_selection,
        default_allowed_llm_selection,
    )

    authorised = apply_allowed_llm_selection(selection)
    if not authorised and not get_current_user().is_admin:
        authorised = default_allowed_llm_selection()
        if authorised is None:
            raise PermissionError("No LLM model is assigned to your account.")
        # ``default_allowed_llm_selection`` reads the grant, then this second
        # check makes its returned logical id subject to the same exact-pair
        # validation as an explicit selection.
        authorised = apply_allowed_llm_selection(authorised)

    if authorised is None:
        return llm_config_module.get_llm_config()
    return llm_config_from_resolved(resolve_llm_runtime_config(llm_selection=authorised))


@dataclass(frozen=True, slots=True)
class LLMSelectionScopeToken:
    config: Token[LLMConfig | None]
    authority: Token[CurrentUser | None]


def activate_llm_selection(selection: Any) -> tuple[LLMConfig, LLMSelectionScopeToken]:
    """Resolve and install a scoped LLM config for the current async context."""
    from deeptutor.multi_user.context import get_current_user
    from deeptutor.multi_user.model_access import (
        reset_runtime_model_authority,
        set_runtime_model_authority,
    )

    config = resolve_llm_config_for_selection(selection)
    authority_token = set_runtime_model_authority(get_current_user())
    try:
        config_token = llm_config_module.set_scoped_llm_config(config)
    except BaseException:
        reset_runtime_model_authority(authority_token)
        raise
    return config, LLMSelectionScopeToken(config=config_token, authority=authority_token)


def reset_llm_selection(token: LLMSelectionScopeToken | None) -> None:
    if token is not None:
        from deeptutor.multi_user.model_access import reset_runtime_model_authority

        llm_config_module.reset_scoped_llm_config(token.config)
        reset_runtime_model_authority(token.authority)


__all__ = [
    "activate_llm_selection",
    "LLMSelectionScopeToken",
    "llm_config_from_resolved",
    "reset_llm_selection",
    "resolve_llm_config_for_selection",
]
