"""Legacy personal-model compatibility helpers.

Consumer-subscription connectors are deployment-owner-only. Older versions
could leave a managed Codex profile in an ordinary user's catalogue; those
profiles are deliberately ignored rather than merged into runtime state.
``owner_catalog_service`` remains the single location helper used by the
deployment owner's Codex lifecycle.
"""

from __future__ import annotations

from typing import Any

from deeptutor.services.config.model_catalog import ModelCatalogService

from .paths import get_owner_path_service


def owner_catalog_service() -> ModelCatalogService:
    """Model catalog of the account that owns the current scope.

    The Codex lifecycle is gated to the deployment owner before this helper is
    reached. CLI and background contexts continue to resolve their configured
    owner root through the shared path service.
    """
    return ModelCatalogService.get_instance(
        get_owner_path_service().get_settings_file("model_catalog")
    )


def personal_llm_rows() -> list[dict[str, Any]]:
    """Return no legacy per-user subscription models."""
    return []


def merge_personal_llm_profiles(catalog: dict[str, Any]) -> dict[str, Any]:
    """Ignore legacy personal profiles and return ``catalog`` unchanged."""
    return catalog


__all__ = [
    "merge_personal_llm_profiles",
    "owner_catalog_service",
    "personal_llm_rows",
]
