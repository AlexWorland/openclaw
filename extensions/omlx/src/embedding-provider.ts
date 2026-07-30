// Omlx provider module implements model/runtime integration.
import { createSubsystemLogger } from "openclaw/plugin-sdk/logging-core";
import {
  buildRemoteBaseUrlPolicy,
  createRemoteEmbeddingProvider,
  normalizeEmbeddingModelWithPrefixes,
  type MemoryEmbeddingProvider,
  type MemoryEmbeddingProviderCreateOptions,
} from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";
import { resolveMemorySecretInputString } from "openclaw/plugin-sdk/memory-core-host-secret";
import { normalizeProviderId } from "openclaw/plugin-sdk/provider-model-shared";
import { formatErrorMessage, type SsrFPolicy } from "openclaw/plugin-sdk/ssrf-runtime";
import { OMLX_DEFAULT_EMBEDDING_MODEL, OMLX_PROVIDER_ID } from "./defaults.js";
import { ensureOmlxModelLoaded } from "./models.fetch.js";
import { resolveOmlxInferenceBase, resolveOmlxServerBase } from "./models.js";
import {
  buildOmlxAuthHeaders,
  resolveOmlxConfiguredApiKeyForProvider,
  resolveOmlxProviderHeaders,
  resolveOmlxRuntimeApiKey,
} from "./runtime.js";

const log = createSubsystemLogger("memory/embeddings");

type OmlxEmbeddingClient = {
  baseUrl: string;
  headers: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  model: string;
};
type MemoryCoreAcquireLocalService = (
  target: {
    providerId: string;
    baseUrl: string;
    headers?: HeadersInit;
  },
  signal?: AbortSignal | null,
) => Promise<{ release: () => void } | undefined>;
type LocalServiceAwareEmbeddingOptions = MemoryEmbeddingProviderCreateOptions & {
  acquireLocalService?: MemoryCoreAcquireLocalService;
};
export const DEFAULT_OMLX_EMBEDDING_MODEL = OMLX_DEFAULT_EMBEDDING_MODEL;

/** Normalizes oMLX embedding model refs and accepts an `omlx/` prefix. */
function normalizeOmlxModel(model: string, providerId?: string): string {
  return normalizeEmbeddingModelWithPrefixes({
    model,
    defaultModel: DEFAULT_OMLX_EMBEDDING_MODEL,
    prefixes: [`${providerId?.trim() || OMLX_PROVIDER_ID}/`, `${OMLX_PROVIDER_ID}/`],
  });
}

function hasAuthorizationHeader(headers: Record<string, string> | undefined): boolean {
  if (!headers) {
    return false;
  }
  return Object.entries(headers).some(
    ([headerName, value]) =>
      headerName.trim().toLowerCase() === "authorization" && value.trim().length > 0,
  );
}

/** Resolves API key (real or synthetic placeholder) from runtime/provider auth config. */
async function resolveOmlxApiKey(
  options: MemoryEmbeddingProviderCreateOptions,
  providerId?: string,
): Promise<string | undefined> {
  const selectedProviderId = providerId?.trim();
  const selectedApiKey =
    selectedProviderId && selectedProviderId !== OMLX_PROVIDER_ID
      ? options.config.models?.providers?.[selectedProviderId]?.apiKey
      : undefined;
  if (selectedProviderId && selectedProviderId !== OMLX_PROVIDER_ID) {
    return selectedApiKey === undefined || selectedApiKey === null
      ? undefined
      : await resolveOmlxConfiguredApiKeyForProvider({
          providerId: selectedProviderId,
          config: options.config,
          env: process.env,
        });
  }
  try {
    return await resolveOmlxRuntimeApiKey({
      config: options.config,
      agentDir: options.agentDir,
    });
  } catch (error) {
    // Embeddings can target local oMLX instances that do not require auth.
    if (/oMLX API key is required/i.test(formatErrorMessage(error))) {
      return undefined;
    }
    throw error;
  }
}

function resolveConfiguredOmlxProvider(options: MemoryEmbeddingProviderCreateOptions) {
  const providers = options.config.models?.providers;
  if (!providers) {
    return undefined;
  }
  const providerId = options.provider?.trim() || OMLX_PROVIDER_ID;
  const direct = providers[providerId];
  if (direct) {
    return { providerId, config: direct };
  }
  const normalized = normalizeProviderId(providerId);
  for (const [candidateId, candidate] of Object.entries(providers)) {
    if (normalizeProviderId(candidateId) === normalized) {
      return { providerId: candidateId, config: candidate };
    }
  }
  const fallback = providers[OMLX_PROVIDER_ID];
  return fallback ? { providerId: OMLX_PROVIDER_ID, config: fallback } : undefined;
}

function resolveOmlxLocalServiceBaseUrl(
  configuredBaseUrl: string | undefined,
  inferenceBaseUrl: string,
): string {
  const configured = configuredBaseUrl?.trim();
  if (!configured) {
    return inferenceBaseUrl;
  }
  return `${resolveOmlxServerBase(configured)}/v1`;
}

/** Creates the oMLX embedding provider client and preloads the target model before return. */
export async function createOmlxEmbeddingProvider(
  options: MemoryEmbeddingProviderCreateOptions,
): Promise<{ provider: MemoryEmbeddingProvider; client: OmlxEmbeddingClient }> {
  const resolvedProvider = resolveConfiguredOmlxProvider(options);
  const providerConfig = resolvedProvider?.config;
  const providerBaseUrl = providerConfig?.baseUrl?.trim();
  const isFallbackActivation = options.fallback === "omlx" && options.provider !== "omlx";
  const remoteBaseUrl = options.remote?.baseUrl?.trim();
  const remoteApiKey = !isFallbackActivation
    ? resolveMemorySecretInputString({
        value: options.remote?.apiKey,
        path: "memory.search.remote.apiKey",
      })
    : undefined;
  // memorySearch.remote is shared across primary + fallback providers.
  // Ignore it during fallback activation to avoid inheriting another provider's
  // endpoint/headers/credentials when oMLX activates as a fallback.
  const baseUrlSource = !isFallbackActivation ? remoteBaseUrl : undefined;
  const configuredBaseUrl =
    baseUrlSource && baseUrlSource.length > 0
      ? baseUrlSource
      : providerBaseUrl && providerBaseUrl.length > 0
        ? providerBaseUrl
        : undefined;
  const baseUrl = resolveOmlxInferenceBase(configuredBaseUrl);
  const model = normalizeOmlxModel(options.model, resolvedProvider?.providerId);
  const providerHeaders = await resolveOmlxProviderHeaders({
    config: options.config,
    env: process.env,
    headers: Object.assign(
      {},
      providerConfig?.headers,
      !isFallbackActivation ? options.remote?.headers : {},
    ),
  });
  const apiKey = hasAuthorizationHeader(providerHeaders)
    ? undefined
    : !isFallbackActivation
      ? remoteApiKey || (await resolveOmlxApiKey(options, resolvedProvider?.providerId))
      : await resolveOmlxApiKey(options, resolvedProvider?.providerId);
  const headerOverrides = Object.assign({}, providerHeaders);
  const headers =
    buildOmlxAuthHeaders({
      apiKey,
      json: true,
      headers: headerOverrides,
    }) ?? {};
  const ssrfPolicy = buildRemoteBaseUrlPolicy(baseUrl);
  const client: OmlxEmbeddingClient = {
    baseUrl,
    model,
    headers,
    ssrfPolicy,
  };
  const localServiceTarget =
    providerConfig?.localService && !baseUrlSource
      ? {
          providerId: resolvedProvider?.providerId ?? OMLX_PROVIDER_ID,
          baseUrl: resolveOmlxLocalServiceBaseUrl(providerBaseUrl, baseUrl),
          headers,
        }
      : undefined;
  const acquireLocalService = (options as LocalServiceAwareEmbeddingOptions).acquireLocalService;
  const withLocalServiceLease = async <T>(
    signal: AbortSignal | undefined,
    action: () => Promise<T>,
  ): Promise<T> => {
    const lease =
      localServiceTarget && acquireLocalService
        ? await acquireLocalService(localServiceTarget, signal)
        : undefined;
    try {
      return await action();
    } finally {
      lease?.release();
    }
  };

  await withLocalServiceLease(undefined, async () => {
    try {
      await ensureOmlxModelLoaded({
        baseUrl,
        apiKey,
        headers: headerOverrides,
        ssrfPolicy,
        modelId: model,
        timeoutMs: 120_000,
      });
    } catch (error) {
      log.warn("oMLX embeddings warmup failed; continuing without preload", {
        baseUrl,
        model,
        error: formatErrorMessage(error),
      });
    }
  });

  const remoteProvider = createRemoteEmbeddingProvider({
    id: OMLX_PROVIDER_ID,
    client,
    errorPrefix: "oMLX embeddings failed",
  });
  const provider: MemoryEmbeddingProvider = {
    ...remoteProvider,
    embedQuery: async (text, callOptions) =>
      await withLocalServiceLease(callOptions?.signal, async () => {
        return await remoteProvider.embedQuery(text, callOptions);
      }),
    embedBatch: async (texts, callOptions) =>
      await withLocalServiceLease(callOptions?.signal, async () => {
        return await remoteProvider.embedBatch(texts, callOptions);
      }),
    ...(remoteProvider.embedBatchInputs
      ? {
          embedBatchInputs: async (
            inputs: Parameters<NonNullable<MemoryEmbeddingProvider["embedBatchInputs"]>>[0],
            callOptions?: Parameters<NonNullable<MemoryEmbeddingProvider["embedBatchInputs"]>>[1],
          ) =>
            await withLocalServiceLease(callOptions?.signal, async () => {
              return await remoteProvider.embedBatchInputs!(inputs, callOptions);
            }),
        }
      : {}),
  };
  return {
    provider,
    client,
  };
}
