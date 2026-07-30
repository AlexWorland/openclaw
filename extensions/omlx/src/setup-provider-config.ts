// Omlx setup module builds the persisted provider config shape.
import type { OpenClawConfig } from "openclaw/plugin-sdk/provider-auth";
import type {
  ModelDefinitionConfig,
  ModelProviderConfig,
} from "openclaw/plugin-sdk/provider-model-shared";
import { withAgentModelAliases } from "openclaw/plugin-sdk/provider-onboard";
import { normalizeStringEntries } from "openclaw/plugin-sdk/string-coerce-runtime";
import { OMLX_LOCAL_API_KEY_PLACEHOLDER, OMLX_PROVIDER_ID as PROVIDER_ID } from "./defaults.js";
import { resolveOmlxProviderAuthMode, shouldUseOmlxApiKeyPlaceholder } from "./provider-auth.js";

/** Drops stored oMLX auth profiles/order so header-only auth does not keep a stale credential. */
export function stripOmlxStoredAuthConfig(cfg: OpenClawConfig): OpenClawConfig {
  const { profiles: _profiles, order: _order, ...restAuth } = cfg.auth ?? {};
  const nextProfiles = Object.fromEntries(
    Object.entries(cfg.auth?.profiles ?? {}).filter(
      ([, profile]) => profile.provider !== PROVIDER_ID,
    ),
  );
  const nextOrder = Object.fromEntries(
    Object.entries(cfg.auth?.order ?? {}).filter(([providerId]) => providerId !== PROVIDER_ID),
  );
  return {
    ...cfg,
    auth:
      Object.keys(restAuth).length > 0 ||
      Object.keys(nextProfiles).length > 0 ||
      Object.keys(nextOrder).length > 0
        ? {
            ...restAuth,
            ...(Object.keys(nextProfiles).length > 0 ? { profiles: nextProfiles } : {}),
            ...(Object.keys(nextOrder).length > 0 ? { order: nextOrder } : {}),
          }
        : undefined,
  };
}

/** Assembles the provider entry setup persists, layering shared-helper output over existing config. */
export function buildOmlxSetupProviderConfig(params: {
  existingProvider: ModelProviderConfig | undefined;
  sharedProvider?: ModelProviderConfig;
  baseUrl: string;
  apiKey?: ModelProviderConfig["apiKey"];
  headers: ModelProviderConfig["headers"] | undefined;
  models: ModelDefinitionConfig[];
}): ModelProviderConfig {
  const existingWithoutAuth = params.existingProvider
    ? (({ auth: _auth, apiKey: _apiKey, ...rest }) => rest)(params.existingProvider)
    : undefined;
  const sharedWithoutAuth = params.sharedProvider
    ? (({ auth: _auth, apiKey: _apiKey, ...rest }) => rest)(params.sharedProvider)
    : undefined;
  const resolvedAuth = resolveOmlxProviderAuthMode(params.apiKey);
  return {
    ...existingWithoutAuth,
    ...sharedWithoutAuth,
    baseUrl: params.baseUrl,
    api: params.sharedProvider?.api ?? params.existingProvider?.api ?? "openai-completions",
    ...(resolvedAuth ? { auth: resolvedAuth } : {}),
    ...(params.apiKey !== undefined ? { apiKey: params.apiKey } : {}),
    headers: params.headers,
    models: params.models,
  };
}

/**
 * Chooses which apiKey value to persist.
 *
 * Explicit api-key auth keeps a real credential (or the env marker); otherwise a
 * reachable local server with models persists the non-secret local placeholder so
 * runtime auth resolution has something to match without inventing a credential.
 */
export function resolvePersistedOmlxApiKey(params: {
  currentApiKey: ModelProviderConfig["apiKey"] | undefined;
  explicitAuth: ModelProviderConfig["auth"] | undefined;
  fallbackApiKey: ModelProviderConfig["apiKey"] | undefined;
  preferFallbackApiKey?: boolean;
  hasModels: boolean;
  hasAuthorizationHeader?: boolean;
}): ModelProviderConfig["apiKey"] | undefined {
  if (params.explicitAuth === "api-key") {
    if (params.preferFallbackApiKey && params.fallbackApiKey !== undefined) {
      return params.fallbackApiKey;
    }
    if (resolveOmlxProviderAuthMode(params.currentApiKey)) {
      return params.currentApiKey;
    }
    return params.fallbackApiKey;
  }
  return shouldUseOmlxApiKeyPlaceholder({
    hasModels: params.hasModels,
    resolvedApiKey: params.currentApiKey,
    hasAuthorizationHeader: params.hasAuthorizationHeader,
  })
    ? OMLX_LOCAL_API_KEY_PLACEHOLDER
    : undefined;
}

/** Keeps explicit model entries first and appends unique discovered entries. */
export function mergeDiscoveredModels(params: {
  explicitModels?: ModelDefinitionConfig[];
  discoveredModels?: ModelDefinitionConfig[];
}): ModelDefinitionConfig[] {
  const explicitModels = Array.isArray(params.explicitModels) ? params.explicitModels : [];
  const discoveredModels = Array.isArray(params.discoveredModels) ? params.discoveredModels : [];
  if (explicitModels.length === 0) {
    return discoveredModels;
  }
  if (discoveredModels.length === 0) {
    return explicitModels;
  }

  const merged = [...explicitModels];
  const seen = new Set(normalizeStringEntries(explicitModels.map((model) => model.id)));
  for (const model of discoveredModels) {
    const id = model.id.trim();
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    merged.push(model);
  }
  return merged;
}

/** Preserves existing allowlist metadata and appends discovered oMLX model refs. */
export function mergeDiscoveredOmlxAllowlistEntries(params: {
  existing?: NonNullable<NonNullable<OpenClawConfig["agents"]>["defaults"]>["models"];
  discoveredModels: ModelDefinitionConfig[];
}) {
  return withAgentModelAliases(
    params.existing,
    normalizeStringEntries(params.discoveredModels.map((model) => model.id)).map(
      (id) => `${PROVIDER_ID}/${id}`,
    ),
  );
}
