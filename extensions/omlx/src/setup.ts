// Omlx setup module handles plugin onboarding behavior.
import type { ProviderAppGuidedSetupContext } from "openclaw/plugin-sdk/plugin-entry";
import {
  removeProviderAuthProfilesWithLock,
  buildApiKeyCredential,
  ensureApiKeyFromEnvOrPrompt,
  hasConfiguredSecretInput,
  normalizeOptionalSecretInput,
  type OpenClawConfig,
  type SecretInput,
  type SecretInputMode,
} from "openclaw/plugin-sdk/provider-auth";
import type { ModelProviderConfig } from "openclaw/plugin-sdk/provider-model-shared";
import {
  applyProviderDefaultModel,
  configureOpenAICompatibleSelfHostedProviderNonInteractive,
  type ProviderAuthMethodNonInteractiveContext,
  type ProviderAuthResult,
  type ProviderCatalogContext,
  type ProviderPrepareDynamicModelContext,
  type ProviderRuntimeModel,
} from "openclaw/plugin-sdk/provider-setup";
import { isTruthyEnvValue } from "openclaw/plugin-sdk/runtime-env";
import { WizardCancelledError, type WizardPrompter } from "openclaw/plugin-sdk/setup";
import {
  OMLX_DEFAULT_API_KEY_ENV_VAR,
  OMLX_DEFAULT_BASE_URL,
  OMLX_DEFAULT_INFERENCE_BASE_URL,
  OMLX_DOCKER_HOST_BASE_URL,
  OMLX_DOCKER_HOST_INFERENCE_BASE_URL,
  OMLX_LOCAL_API_KEY_PLACEHOLDER,
  OMLX_MODEL_PLACEHOLDER,
  OMLX_PROVIDER_LABEL,
  OMLX_PROVIDER_ID as PROVIDER_ID,
} from "./defaults.js";
import { discoverOmlxModels } from "./models.fetch.js";
import { resolveOmlxInferenceBase } from "./models.js";
import {
  hasOmlxAuthorizationHeader,
  resolveOmlxProviderAuthMode,
  shouldUseOmlxApiKeyPlaceholder,
} from "./provider-auth.js";
import {
  resolveOmlxConfiguredApiKey,
  resolveOmlxProviderHeaders,
  resolveOmlxRequestContext,
} from "./runtime.js";
import {
  collectAppGuidedOmlxModelIds,
  discoverOmlxProviderCatalog,
  discoverOmlxSetupModels,
  isOmlxDiscoveryConfigResolutionError,
  selectDefaultOmlxModelId,
} from "./setup-discovery.js";
import {
  buildOmlxSetupProviderConfig,
  mergeDiscoveredModels,
  mergeDiscoveredOmlxAllowlistEntries,
  resolvePersistedOmlxApiKey,
  stripOmlxStoredAuthConfig,
} from "./setup-provider-config.js";

type ProviderPromptText = (params: {
  message: string;
  initialValue?: string;
  placeholder?: string;
  validate?: (value: string | undefined) => string | undefined;
}) => Promise<string | undefined>;

type ProviderPromptNote = (message: string, title?: string) => Promise<void> | void;

function resolveOmlxSetupDefaultBaseUrl(env: NodeJS.ProcessEnv = process.env): string {
  return isTruthyEnvValue(env.OPENCLAW_DOCKER_SETUP)
    ? OMLX_DOCKER_HOST_BASE_URL
    : OMLX_DEFAULT_BASE_URL;
}

function resolveOmlxSetupDefaultInferenceBaseUrl(env: NodeJS.ProcessEnv = process.env): string {
  return isTruthyEnvValue(env.OPENCLAW_DOCKER_SETUP)
    ? OMLX_DOCKER_HOST_INFERENCE_BASE_URL
    : OMLX_DEFAULT_INFERENCE_BASE_URL;
}

/** Read-only local discovery plus a success-gated config proposal for guided setup. */
export async function prepareAppGuidedOmlxSetup(
  ctx: ProviderAppGuidedSetupContext & { modelRef?: string },
): Promise<ProviderAuthResult | null> {
  const existingProvider = ctx.config.models?.providers?.[PROVIDER_ID];
  const baseUrl = resolveOmlxInferenceBase(
    existingProvider?.baseUrl ?? resolveOmlxSetupDefaultInferenceBaseUrl(ctx.env),
  );
  let headers: Record<string, string> | undefined;
  let configuredValue: string | undefined;
  try {
    headers = await resolveOmlxProviderHeaders({
      config: ctx.config,
      env: ctx.env,
      headers: existingProvider?.headers,
    });
    configuredValue = await resolveOmlxConfiguredApiKey({
      config: ctx.config,
      env: ctx.env,
      allowUnresolved: true,
    });
  } catch {
    return null;
  }
  const environmentValue = ctx.env[OMLX_DEFAULT_API_KEY_ENV_VAR]?.trim();
  const accessValue = configuredValue ?? environmentValue;
  const setupDiscovery = await discoverOmlxSetupModels({
    baseUrl,
    apiKey: accessValue ?? OMLX_LOCAL_API_KEY_PLACEHOLDER,
    ...(headers ? { headers } : {}),
    timeoutMs: 5000,
  });
  if ("failure" in setupDiscovery) {
    return null;
  }
  const requestedPrefix = `${PROVIDER_ID}/`;
  const requestedModelId = ctx.modelRef?.startsWith(requestedPrefix)
    ? ctx.modelRef.slice(requestedPrefix.length)
    : undefined;
  const appGuidedModelIds = collectAppGuidedOmlxModelIds(setupDiscovery.value.discovery);
  const selectedModelId =
    requestedModelId ??
    selectDefaultOmlxModelId(
      setupDiscovery.value.models.filter((model) => appGuidedModelIds.has(model.id)),
    );
  if (
    !selectedModelId ||
    !appGuidedModelIds.has(selectedModelId) ||
    !setupDiscovery.value.models.some((model) => model.id === selectedModelId)
  ) {
    return null;
  }
  const persistedAccess = accessValue
    ? (existingProvider?.apiKey ?? OMLX_DEFAULT_API_KEY_ENV_VAR)
    : shouldUseOmlxApiKeyPlaceholder({
          hasModels: true,
          resolvedApiKey: undefined,
          hasAuthorizationHeader: hasOmlxAuthorizationHeader(headers),
        })
      ? OMLX_LOCAL_API_KEY_PLACEHOLDER
      : undefined;
  return {
    profiles: [],
    defaultModel: `${PROVIDER_ID}/${selectedModelId}`,
    configPatch: {
      models: {
        mode: ctx.config.models?.mode ?? "merge",
        providers: {
          [PROVIDER_ID]: buildOmlxSetupProviderConfig({
            apiKey: persistedAccess,
            existingProvider,
            baseUrl,
            headers: existingProvider?.headers,
            models: setupDiscovery.value.models,
          }),
        },
      },
    },
  };
}

/** Interactive oMLX setup with connectivity and model-availability checks. */
export async function promptAndConfigureOmlxInteractive(params: {
  config: OpenClawConfig;
  agentDir?: string;
  prompter?: WizardPrompter;
  secretInputMode?: SecretInputMode;
  allowSecretRefPrompt?: boolean;
  promptText?: ProviderPromptText;
  note?: ProviderPromptNote;
}): Promise<ProviderAuthResult> {
  const promptText = params.prompter
    ? params.prompter.text.bind(params.prompter)
    : params.promptText;
  if (!promptText) {
    throw new Error("oMLX interactive setup requires a text prompter.");
  }
  const note = params.prompter ? params.prompter.note.bind(params.prompter) : params.note;
  const defaultBaseUrl = resolveOmlxSetupDefaultBaseUrl();
  const baseUrlRaw = await promptText({
    message: `${OMLX_PROVIDER_LABEL} base URL`,
    initialValue: defaultBaseUrl,
    placeholder: defaultBaseUrl,
    validate: (value) => (value?.trim() ? undefined : "Required"),
  });
  const baseUrl = resolveOmlxInferenceBase(baseUrlRaw ?? defaultBaseUrl);
  let credentialInput: SecretInput | undefined;
  let credentialMode: SecretInputMode | undefined;
  const implicitRefMode = params.allowSecretRefPrompt === false && !params.secretInputMode;
  const autoRefEnvKey = process.env[OMLX_DEFAULT_API_KEY_ENV_VAR]?.trim();
  const apiKey =
    params.prompter && implicitRefMode && autoRefEnvKey
      ? autoRefEnvKey
      : params.prompter
        ? await ensureApiKeyFromEnvOrPrompt({
            config: params.config,
            provider: PROVIDER_ID,
            envLabel: OMLX_DEFAULT_API_KEY_ENV_VAR,
            promptMessage: `${OMLX_PROVIDER_LABEL} API key`,
            normalize: (value) => value.trim(),
            validate: () => undefined,
            prompter: params.prompter,
            secretInputMode:
              params.allowSecretRefPrompt === false
                ? (params.secretInputMode ?? "plaintext")
                : params.secretInputMode,
            setCredential: async (apiKeyValue, mode) => {
              credentialInput = apiKeyValue;
              credentialMode = mode;
            },
          })
        : (
            (await promptText({
              message: `${OMLX_PROVIDER_LABEL} API key`,
              placeholder: "leave blank if auth is disabled",
              validate: () => undefined,
            })) ?? ""
          ).trim();
  const normalizedApiKey = normalizeOptionalSecretInput(apiKey);
  const credentialSource =
    credentialInput ??
    (implicitRefMode && autoRefEnvKey ? `\${${OMLX_DEFAULT_API_KEY_ENV_VAR}}` : apiKey);
  const shouldStoreCredential = params.prompter
    ? credentialMode === "ref" || hasConfiguredSecretInput(credentialSource)
    : normalizedApiKey !== undefined;
  const credential = shouldStoreCredential
    ? params.prompter
      ? buildApiKeyCredential(
          PROVIDER_ID,
          credentialSource,
          undefined,
          credentialMode
            ? { secretInputMode: credentialMode }
            : implicitRefMode && autoRefEnvKey
              ? { secretInputMode: "ref" }
              : undefined,
        )
      : {
          type: "api_key" as const,
          provider: PROVIDER_ID,
          key: normalizedApiKey ?? apiKey,
        }
    : undefined;
  const existingProvider = params.config.models?.providers?.[PROVIDER_ID];
  // Auth setup updates auth/profile/provider model fields but does not mutate
  // user-provided header overrides. Runtime request assembly is the source of truth for auth.
  const persistedHeaders = existingProvider?.headers;
  const resolvedHeaders = await resolveOmlxProviderHeaders({
    config: params.config,
    env: process.env,
    headers: persistedHeaders,
  });
  const hasAuthorizationHeader = hasOmlxAuthorizationHeader(resolvedHeaders);
  const setupDiscoveryApiKey =
    normalizedApiKey ??
    (shouldUseOmlxApiKeyPlaceholder({
      hasModels: true,
      resolvedApiKey: undefined,
      hasAuthorizationHeader,
    })
      ? OMLX_LOCAL_API_KEY_PLACEHOLDER
      : undefined);
  const setupDiscovery = await discoverOmlxSetupModels({
    baseUrl,
    apiKey: setupDiscoveryApiKey,
    ...(resolvedHeaders ? { headers: resolvedHeaders } : {}),
    timeoutMs: 5000,
  });
  if ("failure" in setupDiscovery) {
    await note?.(setupDiscovery.failure.noteLines.join("\n"), "oMLX");
    throw new WizardCancelledError(setupDiscovery.failure.reason);
  }
  const discoveredModels = setupDiscovery.value.models;
  const allowlistEntries = mergeDiscoveredOmlxAllowlistEntries({
    existing: params.config.agents?.defaults?.models,
    discoveredModels,
  });
  const defaultModel = setupDiscovery.value.defaultModel;
  const persistedApiKey =
    resolvePersistedOmlxApiKey({
      currentApiKey: normalizedApiKey ? existingProvider?.apiKey : undefined,
      explicitAuth: resolveOmlxProviderAuthMode(normalizedApiKey),
      fallbackApiKey: normalizedApiKey ? OMLX_DEFAULT_API_KEY_ENV_VAR : undefined,
      preferFallbackApiKey: true,
      hasModels: discoveredModels.length > 0,
      hasAuthorizationHeader,
    }) ?? (normalizedApiKey ? OMLX_DEFAULT_API_KEY_ENV_VAR : undefined);
  if (!credential) {
    await removeProviderAuthProfilesWithLock({
      provider: PROVIDER_ID,
      agentDir: params.agentDir,
    });
  }

  return {
    profiles: credential
      ? [
          {
            profileId: `${PROVIDER_ID}:default`,
            credential,
          },
        ]
      : [],
    configPatch: {
      agents: {
        defaults: {
          models: allowlistEntries,
        },
      },
      models: {
        // Respect existing global mode; self-hosted provider setup should merge by default.
        mode: params.config.models?.mode ?? "merge",
        providers: {
          [PROVIDER_ID]: buildOmlxSetupProviderConfig({
            existingProvider,
            baseUrl,
            apiKey: persistedApiKey,
            headers: persistedHeaders,
            models: discoveredModels,
          }),
        },
      },
    },
    defaultModel,
  };
}

/** Non-interactive setup path backed by the shared self-hosted helper. */
export async function configureOmlxNonInteractive(
  ctx: ProviderAuthMethodNonInteractiveContext,
): Promise<OpenClawConfig | null> {
  const customBaseUrl = normalizeOptionalSecretInput(ctx.opts.customBaseUrl);
  const baseUrl = resolveOmlxInferenceBase(
    customBaseUrl || resolveOmlxSetupDefaultInferenceBaseUrl(),
  );
  const normalizedCtx = customBaseUrl
    ? {
        ...ctx,
        opts: {
          ...ctx.opts,
          customBaseUrl: baseUrl,
        },
      }
    : ctx;
  const requestedModelId = normalizeOptionalSecretInput(normalizedCtx.opts.customModelId);
  const resolved = await normalizedCtx.resolveApiKey({
    provider: PROVIDER_ID,
    flagValue:
      normalizeOptionalSecretInput(normalizedCtx.opts.omlxApiKey) ??
      normalizeOptionalSecretInput(normalizedCtx.opts.customApiKey),
    flagName:
      normalizeOptionalSecretInput(normalizedCtx.opts.omlxApiKey) !== undefined
        ? "--omlx-api-key"
        : "--custom-api-key",
    envVar: OMLX_DEFAULT_API_KEY_ENV_VAR,
    envVarName: OMLX_DEFAULT_API_KEY_ENV_VAR,
    required: false,
  });

  const existingProvider = normalizedCtx.config.models?.providers?.[PROVIDER_ID];
  // Auth setup updates auth/profile/provider model fields but does not mutate
  // user-provided header overrides. Runtime request assembly is the source of truth for auth.
  const persistedHeaders = existingProvider?.headers;
  const resolvedHeaders = await resolveOmlxProviderHeaders({
    config: normalizedCtx.config,
    env: process.env,
    headers: persistedHeaders,
  });
  const hasAuthorizationHeader = hasOmlxAuthorizationHeader(resolvedHeaders);
  const useHeaderOnlyAuth = hasAuthorizationHeader && (!resolved || resolved.source !== "flag");
  // A local oMLX server needs no credential, so an absent key still yields the
  // non-secret local placeholder rather than failing setup.
  const setupDiscoveryApiKey =
    (useHeaderOnlyAuth ? undefined : resolved?.key) ??
    (shouldUseOmlxApiKeyPlaceholder({
      hasModels: true,
      resolvedApiKey: undefined,
      hasAuthorizationHeader,
    })
      ? OMLX_LOCAL_API_KEY_PLACEHOLDER
      : undefined);
  const setupDiscovery = await discoverOmlxSetupModels({
    baseUrl,
    apiKey: setupDiscoveryApiKey,
    ...(resolvedHeaders ? { headers: resolvedHeaders } : {}),
    timeoutMs: 5000,
  });
  if ("failure" in setupDiscovery) {
    normalizedCtx.runtime.error(setupDiscovery.failure.noteLines.join("\n"));
    normalizedCtx.runtime.exit(1);
    return null;
  }
  const discoveredModels = setupDiscovery.value.models;
  const selectedModelId = requestedModelId ?? setupDiscovery.value.defaultModelId;
  const selectedModel = selectedModelId
    ? discoveredModels.find((model) => model.id === selectedModelId)
    : undefined;
  if (!selectedModelId || !selectedModel) {
    const availableModels = discoveredModels.map((model) => model.id).join(", ");
    normalizedCtx.runtime.error(
      requestedModelId
        ? [
            `oMLX model ${requestedModelId} was not found at ${baseUrl}.`,
            `Available models: ${availableModels}`,
          ].join("\n")
        : [
            `oMLX did not expose a usable default model at ${baseUrl}.`,
            `Available models: ${availableModels || "(none)"}`,
          ].join("\n"),
    );
    normalizedCtx.runtime.exit(1);
    return null;
  }
  if (useHeaderOnlyAuth) {
    await removeProviderAuthProfilesWithLock({
      provider: PROVIDER_ID,
      agentDir: normalizedCtx.agentDir,
    });
    const configWithoutStoredOmlxAuth = stripOmlxStoredAuthConfig(normalizedCtx.config);
    return applyProviderDefaultModel(
      {
        ...configWithoutStoredOmlxAuth,
        models: {
          ...configWithoutStoredOmlxAuth.models,
          mode: configWithoutStoredOmlxAuth.models?.mode ?? "merge",
          providers: {
            ...configWithoutStoredOmlxAuth.models?.providers,
            [PROVIDER_ID]: buildOmlxSetupProviderConfig({
              existingProvider,
              baseUrl,
              headers: persistedHeaders,
              models: discoveredModels,
            }),
          },
        },
      },
      `${PROVIDER_ID}/${selectedModelId}`,
    );
  }
  const resolvedOrSynthetic =
    resolved ??
    (setupDiscoveryApiKey
      ? {
          key: setupDiscoveryApiKey,
          source: "flag" as const,
        }
      : null);
  if (!resolvedOrSynthetic) {
    return null;
  }

  // Delegate to the shared helper even when modelId is set so that onboarding
  // state and credential storage are handled consistently. The pre-resolved key
  // is injected via resolveApiKey to skip a second prompt. The returned config
  // is then post-patched below to add the discovered model list and base URL.
  const configured = await configureOpenAICompatibleSelfHostedProviderNonInteractive({
    ctx: {
      ...normalizedCtx,
      opts: {
        ...normalizedCtx.opts,
        customModelId: selectedModelId,
      },
      resolveApiKey: async () => resolvedOrSynthetic,
    },
    providerId: PROVIDER_ID,
    providerLabel: OMLX_PROVIDER_LABEL,
    defaultBaseUrl: resolveOmlxSetupDefaultInferenceBaseUrl(),
    defaultApiKeyEnvVar: OMLX_DEFAULT_API_KEY_ENV_VAR,
    modelPlaceholder: OMLX_MODEL_PLACEHOLDER,
  });
  if (!configured) {
    return null;
  }
  const sharedProvider = configured.models?.providers?.[PROVIDER_ID];
  const resolvedSyntheticLocalKey = resolvedOrSynthetic.key === OMLX_LOCAL_API_KEY_PLACEHOLDER;
  const persistedApiKey = resolvePersistedOmlxApiKey({
    // If this run resolved to keyless local mode, avoid preserving stale env markers.
    currentApiKey: resolvedSyntheticLocalKey ? undefined : existingProvider?.apiKey,
    explicitAuth: resolveOmlxProviderAuthMode(resolvedOrSynthetic.key),
    fallbackApiKey: resolvedSyntheticLocalKey
      ? OMLX_LOCAL_API_KEY_PLACEHOLDER
      : (configured.models?.providers?.[PROVIDER_ID]?.apiKey ?? OMLX_DEFAULT_API_KEY_ENV_VAR),
    preferFallbackApiKey: true,
    hasModels: discoveredModels.length > 0,
    hasAuthorizationHeader: hasOmlxAuthorizationHeader(resolvedHeaders),
  });

  return {
    ...configured,
    models: {
      ...configured.models,
      providers: {
        ...configured.models?.providers,
        [PROVIDER_ID]: buildOmlxSetupProviderConfig({
          existingProvider,
          sharedProvider,
          baseUrl,
          apiKey: persistedApiKey,
          headers: persistedHeaders,
          models: discoveredModels,
        }),
      },
    },
  };
}

/** Discovers provider settings, merging explicit config with live model discovery. */
export async function discoverOmlxProvider(ctx: ProviderCatalogContext): Promise<{
  provider: ModelProviderConfig;
} | null> {
  const explicit = ctx.config.models?.providers?.[PROVIDER_ID];
  const explicitAuth = explicit?.auth;
  let explicitWithoutHeaders: Omit<ModelProviderConfig, "headers" | "auth" | "apiKey"> | undefined;
  if (explicit) {
    const { headers: _headers, auth: _auth, apiKey: _apiKey, ...rest } = explicit;
    explicitWithoutHeaders = rest;
  }
  const hasExplicitModels = Array.isArray(explicit?.models) && explicit.models.length > 0;
  const { apiKey, discoveryApiKey } = ctx.resolveProviderApiKey(PROVIDER_ID);
  let resolvedHeaders: Record<string, string> | undefined;
  try {
    resolvedHeaders = await resolveOmlxProviderHeaders({
      config: ctx.config,
      env: ctx.env,
      headers: explicit?.headers,
    });
  } catch (error) {
    if (isOmlxDiscoveryConfigResolutionError(error)) {
      return null;
    }
    throw error;
  }
  const hasAuthorizationHeader = hasOmlxAuthorizationHeader(resolvedHeaders);
  let configuredDiscoveryApiKey: string | undefined;
  try {
    configuredDiscoveryApiKey = await resolveOmlxConfiguredApiKey({
      config: ctx.config,
      env: ctx.env,
      allowUnresolved: hasAuthorizationHeader || Boolean(discoveryApiKey),
    });
  } catch (error) {
    if (isOmlxDiscoveryConfigResolutionError(error)) {
      return null;
    }
    throw error;
  }
  const resolvedDiscoveryApiKey = hasAuthorizationHeader
    ? undefined
    : (discoveryApiKey ?? configuredDiscoveryApiKey);
  // CLI/runtime-resolved key takes precedence over static provider config key.
  const resolvedApiKey = apiKey ?? explicit?.apiKey;
  if (hasExplicitModels && explicitWithoutHeaders) {
    const persistedApiKey = resolvePersistedOmlxApiKey({
      currentApiKey: resolvedApiKey,
      explicitAuth,
      fallbackApiKey: OMLX_DEFAULT_API_KEY_ENV_VAR,
      hasModels: hasExplicitModels,
      hasAuthorizationHeader,
    });
    const persistedAuth = resolveOmlxProviderAuthMode(persistedApiKey);
    return {
      provider: {
        ...explicitWithoutHeaders,
        ...(resolvedHeaders ? { headers: resolvedHeaders } : {}),
        baseUrl: resolveOmlxInferenceBase(explicitWithoutHeaders.baseUrl),
        // Keep explicit API unless absent, then fall back to provider default.
        api: explicitWithoutHeaders.api ?? "openai-completions",
        ...(persistedApiKey ? { apiKey: persistedApiKey } : {}),
        ...(persistedAuth ? { auth: persistedAuth } : {}),
        models: explicitWithoutHeaders.models,
      },
    };
  }
  const provider = await discoverOmlxProviderCatalog({
    baseUrl: explicit?.baseUrl,
    // Prefer resolved discovery auth, then configured provider auth.
    apiKey: resolvedDiscoveryApiKey,
    headers: resolvedHeaders,
    quiet: !apiKey && !explicit && !resolvedDiscoveryApiKey,
  });
  const models = mergeDiscoveredModels({
    explicitModels: explicit?.models,
    discoveredModels: provider.models,
  });
  if (models.length === 0 && !apiKey && !explicit?.apiKey) {
    return null;
  }
  const persistedApiKey = resolvePersistedOmlxApiKey({
    currentApiKey: resolvedApiKey,
    explicitAuth,
    fallbackApiKey: OMLX_DEFAULT_API_KEY_ENV_VAR,
    hasModels: models.length > 0,
    hasAuthorizationHeader,
  });
  const persistedAuth = resolveOmlxProviderAuthMode(persistedApiKey);
  return {
    provider: {
      ...provider,
      ...explicitWithoutHeaders,
      ...(resolvedHeaders ? { headers: resolvedHeaders } : {}),
      baseUrl: resolveOmlxInferenceBase(explicit?.baseUrl ?? provider.baseUrl),
      ...(persistedApiKey ? { apiKey: persistedApiKey } : {}),
      ...(persistedAuth ? { auth: persistedAuth } : {}),
      models,
    },
  };
}

/** Discovers live runtime models so a configured-but-unlisted model id still resolves. */
export async function prepareOmlxDynamicModels(
  ctx: ProviderPrepareDynamicModelContext,
): Promise<ProviderRuntimeModel[]> {
  const baseUrl = resolveOmlxInferenceBase(ctx.providerConfig?.baseUrl);
  const { apiKey, headers } = await resolveOmlxRequestContext({
    config: ctx.config,
    agentDir: ctx.agentDir,
    env: process.env,
    providerHeaders: ctx.providerConfig?.headers,
  });
  const discoveredModels = await discoverOmlxModels({
    baseUrl,
    apiKey: apiKey ?? "",
    headers,
    quiet: true,
  });
  return discoveredModels.map((model) =>
    Object.assign({}, model, {
      provider: PROVIDER_ID,
      api: ctx.providerConfig?.api ?? `openai-completions`,
      baseUrl,
      input: model.input.filter(
        (entry): entry is "text" | "image" => entry === "text" || entry === "image",
      ),
    }),
  );
}
