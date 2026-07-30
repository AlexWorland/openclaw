// Omlx plugin module implements models.fetch behavior.
import { formatErrorMessage } from "openclaw/plugin-sdk/error-runtime";
import { createSubsystemLogger } from "openclaw/plugin-sdk/logging-core";
import { resolveTimerTimeoutMs } from "openclaw/plugin-sdk/number-runtime";
import {
  readProviderJsonArrayFieldResponse,
  readResponseTextLimited,
} from "openclaw/plugin-sdk/provider-http";
import type { ModelDefinitionConfig } from "openclaw/plugin-sdk/provider-model-shared";
import { SELF_HOSTED_DEFAULT_COST } from "openclaw/plugin-sdk/provider-setup";
import {
  fetchWithSsrFGuard,
  ssrfPolicyFromHttpBaseUrlAllowedHostname,
  type SsrFPolicy,
} from "openclaw/plugin-sdk/ssrf-runtime";
import { OMLX_DISCOVERY_TIMEOUT_MS, OMLX_LOAD_TIMEOUT_MS } from "./defaults.js";
import {
  buildOmlxModelName,
  mapOmlxWireEntry,
  normalizeOmlxModelId,
  resolveOmlxServerBase,
  type OmlxModelWire,
} from "./models.js";
import { buildOmlxAuthHeaders } from "./runtime.js";

const log = createSubsystemLogger("extensions/omlx/models");
const OMLX_ERROR_BODY_LIMIT_BYTES = 8 * 1024;

type FetchOmlxModelsResult = {
  reachable: boolean;
  status?: number;
  models: OmlxModelWire[];
  error?: unknown;
};

type DiscoverOmlxModelsParams = {
  baseUrl: string;
  apiKey: string;
  headers?: Record<string, string>;
  quiet: boolean;
  /** Injectable fetch implementation; defaults to the global fetch. */
  fetchImpl?: typeof fetch;
};

async function cancelUnreadResponseBody(response: Response): Promise<void> {
  if (!response.bodyUsed) {
    await response.body?.cancel().catch(() => undefined);
  }
}

async function fetchOmlxEndpoint(params: {
  url: string;
  init?: RequestInit;
  timeoutMs: number;
  fetchImpl?: typeof fetch;
  ssrfPolicy?: SsrFPolicy;
  auditContext: string;
}): Promise<{ response: Response; release: () => Promise<void> }> {
  const timeoutMs = resolveTimerTimeoutMs(params.timeoutMs, 1);
  const { response, release } = await fetchWithSsrFGuard({
    url: params.url,
    init: params.init,
    timeoutMs,
    fetchImpl: params.fetchImpl,
    // oMLX is a local server, so the guard must trust the configured hostname or
    // every loopback/LAN discovery call fails the private-network check. Scoping
    // the allowlist to that one hostname still blocks redirects elsewhere.
    policy: params.ssrfPolicy ?? ssrfPolicyFromHttpBaseUrlAllowedHostname(params.url),
    auditContext: params.auditContext,
  });
  return {
    response,
    release: async () => {
      await cancelUnreadResponseBody(response);
      await release();
    },
  };
}

/** Fetches `GET {baseUrl}/models/status` and reports transport reachability separately from HTTP status. */
export async function fetchOmlxModels(params: {
  baseUrl?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  timeoutMs?: number;
  /** Injectable fetch implementation; defaults to the global fetch. */
  fetchImpl?: typeof fetch;
}): Promise<FetchOmlxModelsResult> {
  const baseUrl = resolveOmlxServerBase(params.baseUrl);
  const timeoutMs = params.timeoutMs ?? OMLX_DISCOVERY_TIMEOUT_MS;
  try {
    const { response, release } = await fetchOmlxEndpoint({
      url: `${baseUrl}/v1/models/status`,
      init: {
        headers: buildOmlxAuthHeaders({ apiKey: params.apiKey, headers: params.headers }),
      },
      timeoutMs,
      fetchImpl: params.fetchImpl,
      ssrfPolicy: params.ssrfPolicy,
      auditContext: "omlx-model-discovery",
    });
    try {
      if (!response.ok) {
        return { reachable: true, status: response.status, models: [] };
      }
      let models: unknown[];
      try {
        // oMLX wraps the catalog in a status envelope (memory ceiling, counts, then
        // `models`), so read the field rather than expecting a bare array.
        models = await readProviderJsonArrayFieldResponse(response, "oMLX model list", "models");
      } catch (error) {
        // A reachable server that answered with an unexpected body is not a transport
        // failure; keep `reachable` truthful and carry the parse error for the log.
        return { reachable: true, status: response.status, models: [], error };
      }
      const validModels = models.filter(
        (entry): entry is OmlxModelWire =>
          typeof entry === "object" && entry !== null && !Array.isArray(entry),
      );
      return { reachable: true, status: response.status, models: validModels };
    } finally {
      await release();
    }
  } catch (error) {
    return { reachable: false, models: [], error };
  }
}

/** Discovers LLM/VLM models from oMLX and maps them to OpenClaw model definitions. */
export async function discoverOmlxModels(
  params: DiscoverOmlxModelsParams,
): Promise<ModelDefinitionConfig[]> {
  const fetched = await fetchOmlxModels({
    baseUrl: params.baseUrl,
    apiKey: params.apiKey,
    headers: params.headers,
    fetchImpl: params.fetchImpl,
  });
  const quiet = params.quiet;
  if (!fetched.reachable) {
    if (!quiet) {
      log.debug(`Failed to discover oMLX models: ${formatErrorMessage(fetched.error)}`);
    }
    return [];
  }
  if (fetched.status !== undefined && fetched.status >= 400) {
    if (!quiet) {
      log.debug(`Failed to discover oMLX models: ${fetched.status}`);
    }
    return [];
  }
  const models = fetched.models;
  if (models.length === 0) {
    if (!quiet) {
      log.debug(
        fetched.error === undefined
          ? "No oMLX models found on local instance"
          : `Failed to read oMLX model list: ${formatErrorMessage(fetched.error)}`,
      );
    }
    return [];
  }

  return models
    .map((entry): ModelDefinitionConfig | null => {
      const base = mapOmlxWireEntry(entry);
      if (!base) {
        return null;
      }
      return {
        id: base.id,
        // Runtime display: include vision/reasoning/loaded tags in the name.
        name: buildOmlxModelName(base),
        reasoning: base.reasoning,
        input: base.input,
        cost: SELF_HOSTED_DEFAULT_COST,
        compat: { supportsUsageInStreaming: true },
        contextWindow: base.contextWindow,
        contextTokens: base.contextTokens,
        maxTokens: base.maxTokens,
      };
    })
    .filter((entry): entry is ModelDefinitionConfig => entry !== null);
}

/** Ensures a model is loaded in oMLX before first real inference/embedding call. */
export async function ensureOmlxModelLoaded(params: {
  baseUrl?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  modelId: string;
  timeoutMs?: number;
  /** Injectable fetch implementation; defaults to the global fetch. */
  fetchImpl?: typeof fetch;
}): Promise<string> {
  const modelId = normalizeOmlxModelId(params.modelId);
  if (!modelId) {
    throw new Error("oMLX model id is required");
  }

  const timeoutMs = params.timeoutMs ?? OMLX_LOAD_TIMEOUT_MS;
  const baseUrl = resolveOmlxServerBase(params.baseUrl);
  const preflight = await fetchOmlxModels({
    baseUrl,
    apiKey: params.apiKey,
    headers: params.headers,
    ssrfPolicy: params.ssrfPolicy,
    timeoutMs,
    fetchImpl: params.fetchImpl,
  });
  if (!preflight.reachable) {
    throw new Error(`oMLX model discovery failed: ${formatErrorMessage(preflight.error)}`);
  }
  if (preflight.status !== undefined && preflight.status >= 400) {
    throw new Error(`oMLX model discovery failed (${preflight.status})`);
  }
  const matchingModel = preflight.models.find((entry) => entry.id === modelId);
  if (matchingModel?.loaded === true) {
    return modelId;
  }

  try {
    const { response, release } = await fetchOmlxEndpoint({
      url: `${baseUrl}/v1/models/${encodeURIComponent(modelId)}/load`,
      init: {
        method: "POST",
        headers: buildOmlxAuthHeaders({ apiKey: params.apiKey, headers: params.headers }),
      },
      timeoutMs,
      fetchImpl: params.fetchImpl,
      ssrfPolicy: params.ssrfPolicy,
      auditContext: "omlx-model-load",
    });
    try {
      if (!response.ok) {
        const body = await readResponseTextLimited(response, OMLX_ERROR_BODY_LIMIT_BYTES);
        throw new Error(`oMLX model load failed (${response.status})${body ? `: ${body}` : ""}`);
      }
    } finally {
      await release();
    }
  } catch (error) {
    throw error instanceof Error ? error : new Error(String(error), { cause: error });
  }
  return modelId;
}
