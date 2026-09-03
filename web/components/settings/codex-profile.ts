import type {
  CatalogProfile,
  ProviderOption,
  ServiceName,
} from "./SettingsContext";

/** The tag the backend stamps on the profile its Codex OAuth service owns. */
export const CODEX_MANAGED_BY = "openai_codex_oauth";

/** Profiles written by the Codex OAuth service are read-only in the settings form. */
export function isManagedCodexProfile(
  profile: Pick<CatalogProfile, "managed_by"> | null | undefined,
): boolean {
  return profile?.managed_by === CODEX_MANAGED_BY;
}

/** A managed profile may accept reasoning overrides only after an account-bound refresh. */
export function isBoundManagedCodexProfile(
  profile:
    | Pick<CatalogProfile, "managed_by" | "codex_account_binding">
    | null
    | undefined,
): boolean {
  return (
    isManagedCodexProfile(profile) &&
    typeof profile?.codex_account_binding === "string" &&
    Boolean(profile.codex_account_binding.trim())
  );
}

/** Connection fields consumed by the subscription-backed LLM providers. */
export function subscriptionProviderFields(
  service: ServiceName,
  providerValue: string,
): { apiKey: boolean; baseUrl: boolean; baseUrlRequired: boolean } | null {
  if (service !== "llm") return null;
  if (providerValue === "cursor_subscription") {
    return { apiKey: true, baseUrl: false, baseUrlRequired: false };
  }
  if (providerValue === "grok_subscription") {
    return { apiKey: false, baseUrl: false, baseUrlRequired: false };
  }
  return null;
}

export type SubscriptionProviderModel = {
  name: string;
  model: string;
};

/** The single model exposed by each owner-bound subscription integration. */
export function subscriptionProviderDefaultModel(
  providerValue: string,
): SubscriptionProviderModel | null {
  if (providerValue === "cursor_subscription") {
    return {
      name: "Grok 4.6 High (Cursor Ultra)",
      model: "cursor-grok-4.6-high",
    };
  }
  if (providerValue === "grok_subscription") {
    return {
      name: "Grok 4.6 High (SuperGrok)",
      model: "grok-4.6-high",
    };
  }
  return null;
}

/**
 * Whether a profile authenticates through Codex OAuth instead of typed credentials.
 *
 * A managed profile qualifies even before the provider list has loaded, so a
 * reload can never briefly render API-key fields over an OAuth-only profile.
 */
export function isCodexOAuthProfile(
  service: ServiceName,
  providerValue: string,
  providerOption: Pick<ProviderOption, "auth_mode"> | undefined,
  profile: Pick<CatalogProfile, "managed_by"> | null | undefined,
): boolean {
  return (
    service === "llm" &&
    providerValue === "openai_codex" &&
    (providerOption?.auth_mode === "oauth" || isManagedCodexProfile(profile))
  );
}
