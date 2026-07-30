// Omlx setup module runs setup-time model discovery and default-model selection.
import {
  selectPreferredLocalModelId,
  type ModelDefinitionConfig,
  type ModelProviderConfig,
} from "openclaw/plugin-sdk/provider-model-shared";
import { normalizeStringEntries } from "openclaw/plugin-sdk/string-coerce-runtime";
import { OMLX_DEFAULT_MODEL_ID, OMLX_PROVIDER_ID as PROVIDER_ID } from "./defaults.js";
import { discoverOmlxModels, fetchOmlxModels } from "./models.fetch.js";
import {
  mapOmlxWireModelsToConfig,
  resolveOmlxContextWindow,
  resolveOmlxInferenceBase,
  type OmlxModelWire,
} from "./models.js";

export type OmlxDiscoveryResult = Awaited<ReturnType<typeof fetchOmlxModels>>;

export type OmlxSetupDiscovery = {
  discovery: OmlxDiscoveryResult;
  models: ModelDefinitionConfig[];
  defaultModel: string | undefined;
  defaultModelId: string | undefined;
};

export type OmlxSetupDiscoveryFailure = { noteLines: [string, string]; reason: string };

// oMLX has no tool-use capability signal in its discovery payload; app-guided
// selection instead requires a minimum context window so onboarding does not
// silently default to a tiny/toy model.
const OMLX_APP_GUIDED_MIN_CONTEXT_TOKENS = 16_384;

/** Classifies a setup-time discovery response into an actionable operator message. */
export function resolveOmlxDiscoveryFailure(params: {
  baseUrl: string;
  discovery: OmlxDiscoveryResult;
}): OmlxSetupDiscoveryFailure | null {
  const { baseUrl, discovery } = params;
  if (!discovery.reachable) {
    return {
      noteLines: [
        `oMLX could not be reached at ${baseUrl}.`,
        "Start the oMLX server and re-run setup.",
      ],
      reason: "oMLX not reachable",
    };
  }
  if (discovery.status !== undefined && discovery.status >= 400) {
    return {
      noteLines: [
        `oMLX returned HTTP ${discovery.status} while listing models at ${baseUrl}.`,
        "Check the base URL and API key, then re-run setup.",
      ],
      reason: `oMLX discovery failed (${discovery.status})`,
    };
  }
  const hasUsableModel = discovery.models.some(
    (model) =>
      (model.model_type === "llm" || model.model_type === "vlm") &&
      model.is_helper !== true &&
      Boolean(typeof model.id === "string" && model.id.trim()),
  );
  if (!hasUsableModel) {
    return {
      noteLines: [
        `No oMLX LLM/VLM models were found at ${baseUrl}.`,
        "Load at least one model in oMLX, then re-run setup.",
      ],
      reason: "No oMLX models found",
    };
  }
  return null;
}

/** Builds a provider entry from live discovery only, without consulting existing config. */
export async function discoverOmlxProviderCatalog(params: {
  baseUrl?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  quiet: boolean;
}): Promise<ModelProviderConfig> {
  const baseUrl = resolveOmlxInferenceBase(params.baseUrl);
  const models = await discoverOmlxModels({
    baseUrl,
    apiKey: params.apiKey ?? "",
    headers: params.headers,
    quiet: params.quiet,
  });
  return {
    baseUrl,
    api: "openai-completions",
    models,
  };
}

/**
 * Detects unresolved-secret errors raised while reading oMLX provider config.
 *
 * Catalog discovery must stay silent for a config it cannot resolve rather than
 * failing the whole model-catalog build for every other provider.
 */
export function isOmlxDiscoveryConfigResolutionError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes("models.providers.omlx.apiKey") ||
    message.includes("models.providers.omlx.headers.")
  );
}

/** Prefers the bundled default model id, then the shared local-model preference order. */
export function selectDefaultOmlxModelId(
  discoveredModels: ModelDefinitionConfig[],
): string | undefined {
  const ids = normalizeStringEntries(discoveredModels.map((model) => model.id));
  if (ids.length === 0) {
    return undefined;
  }
  return ids.includes(OMLX_DEFAULT_MODEL_ID)
    ? OMLX_DEFAULT_MODEL_ID
    : (selectPreferredLocalModelId(ids) ?? ids[0]);
}

/** Restricts app-guided onboarding to text LLMs with a usable context window. */
export function collectAppGuidedOmlxModelIds(discovery: OmlxDiscoveryResult): Set<string> {
  return new Set(
    discovery.models.flatMap((entry: OmlxModelWire) => {
      const id = typeof entry.id === "string" ? entry.id.trim() : "";
      if (entry.model_type !== "llm" || entry.is_helper === true || !id) {
        return [];
      }
      const contextWindow = resolveOmlxContextWindow(entry);
      return contextWindow >= OMLX_APP_GUIDED_MIN_CONTEXT_TOKENS ? [id] : [];
    }),
  );
}

/** Fetches setup-time discovery and maps it, or returns the operator-facing failure. */
export async function discoverOmlxSetupModels(params: {
  baseUrl: string;
  apiKey?: string;
  headers?: Record<string, string>;
  timeoutMs?: number;
}): Promise<{ value: OmlxSetupDiscovery } | { failure: OmlxSetupDiscoveryFailure }> {
  const discovery = await fetchOmlxModels({
    baseUrl: params.baseUrl,
    apiKey: params.apiKey,
    ...(params.headers ? { headers: params.headers } : {}),
    timeoutMs: params.timeoutMs ?? 5000,
  });
  const failure = resolveOmlxDiscoveryFailure({ baseUrl: params.baseUrl, discovery });
  if (failure) {
    return { failure };
  }
  const models = mapOmlxWireModelsToConfig(discovery.models);
  const defaultModelId = selectDefaultOmlxModelId(models);
  return {
    value: {
      discovery,
      models,
      defaultModel: defaultModelId ? `${PROVIDER_ID}/${defaultModelId}` : undefined,
      defaultModelId,
    },
  };
}
