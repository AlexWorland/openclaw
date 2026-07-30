// Omlx plugin module implements models behavior.
import {
  DEFAULT_CONTEXT_TOKENS,
  type ModelDefinitionConfig,
  type ModelProviderConfig,
} from "openclaw/plugin-sdk/provider-model-shared";
import {
  SELF_HOSTED_DEFAULT_CONTEXT_WINDOW,
  SELF_HOSTED_DEFAULT_COST,
  SELF_HOSTED_DEFAULT_MAX_TOKENS,
} from "openclaw/plugin-sdk/provider-setup";
import { asPositiveSafeInteger, uniqueStrings } from "openclaw/plugin-sdk/string-coerce-runtime";
import { OMLX_DEFAULT_BASE_URL } from "./defaults.js";

/** A single entry from oMLX's `GET /v1/models/status` discovery response. */
export type OmlxModelWire = {
  id?: unknown;
  model_type?: unknown;
  engine_type?: unknown;
  max_context_window?: unknown;
  model_context_length?: unknown;
  max_tokens?: unknown;
  thinking_default?: unknown;
  is_helper?: unknown;
  loaded?: unknown;
  source_repo_id?: unknown;
};

const OMLX_CHAT_MODEL_TYPES = new Set(["llm", "vlm"]);

type OmlxConfiguredCatalogEntry = {
  id: string;
  name?: string;
  contextWindow?: number;
  contextTokens?: number;
  reasoning?: boolean;
  input?: ("text" | "image" | "document")[];
  compat?: ModelDefinitionConfig["compat"];
};

/** Resolves the effective context window: `max_context_window` first, then `model_context_length`. */
export function resolveOmlxContextWindow(
  entry: Pick<OmlxModelWire, "max_context_window" | "model_context_length">,
): number {
  return (
    asPositiveSafeInteger(entry.max_context_window) ??
    asPositiveSafeInteger(entry.model_context_length) ??
    SELF_HOSTED_DEFAULT_CONTEXT_WINDOW
  );
}

/** Strips leading/trailing slashes so ids are safe to embed in load/unload URLs. */
export function normalizeOmlxModelId(id: string): string {
  return id.trim().replace(/^\/+|\/+$/g, "");
}

function normalizeUrlPath(pathname: string): string {
  const trimmed = pathname.replace(/\/+$/, "");
  if (!trimmed) {
    return "";
  }
  return trimmed.replace(/\/v1$/i, "");
}

function hasExplicitHttpScheme(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function isLikelyHostBaseUrl(value: string): boolean {
  return (
    /^(?:localhost|(?:\d{1,3}\.){3}\d{1,3}|[a-z0-9.-]+\.[a-z]{2,}|[^/\s?#]+:\d+)(?:[/?#].*)?$/i.test(
      value,
    ) && !value.startsWith("/")
  );
}

function toFetchableOmlxBaseUrl(value: string): string {
  if (hasExplicitHttpScheme(value) || !isLikelyHostBaseUrl(value)) {
    return value;
  }
  return `http://${value}`;
}

/** Resolves oMLX server base URL (without /v1). */
export function resolveOmlxServerBase(configuredBaseUrl?: string): string {
  // Use configured value when present; otherwise target local oMLX default.
  const configured = configuredBaseUrl?.trim();
  const resolved = configured && configured.length > 0 ? configured : OMLX_DEFAULT_BASE_URL;
  const fetchableBaseUrl = toFetchableOmlxBaseUrl(resolved);
  try {
    const parsed = new URL(fetchableBaseUrl);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new TypeError(`Unsupported oMLX protocol: ${parsed.protocol}`);
    }
    const pathname = normalizeUrlPath(parsed.pathname);
    parsed.pathname = pathname.length > 0 ? pathname : "/";
    parsed.search = "";
    parsed.hash = "";
    return parsed.toString().replace(/\/$/, "");
  } catch {
    const trimmed = resolved.replace(/\/+$/, "");
    const normalized = normalizeUrlPath(trimmed);
    return normalized.length > 0 ? normalized : OMLX_DEFAULT_BASE_URL;
  }
}

/** Resolves oMLX inference base URL and always appends /v1. */
export function resolveOmlxInferenceBase(configuredBaseUrl?: string): string {
  const serverBase = resolveOmlxServerBase(configuredBaseUrl);
  return `${serverBase}/v1`;
}

/** Canonicalizes persisted oMLX provider config to the inference base URL form. */
export function normalizeOmlxProviderConfig(provider: ModelProviderConfig): ModelProviderConfig {
  const configuredBaseUrl = typeof provider.baseUrl === "string" ? provider.baseUrl.trim() : "";
  if (!configuredBaseUrl) {
    return provider;
  }
  const normalizedBaseUrl = resolveOmlxInferenceBase(configuredBaseUrl);
  const request =
    provider.request && typeof provider.request === "object" && !Array.isArray(provider.request)
      ? provider.request
      : undefined;
  const requestWithPrivateNetworkDefault =
    typeof request?.allowPrivateNetwork === "boolean"
      ? request
      : {
          ...request,
          allowPrivateNetwork: true,
        };
  if (
    normalizedBaseUrl === provider.baseUrl &&
    requestWithPrivateNetworkDefault === provider.request
  ) {
    return provider;
  }
  return {
    ...provider,
    baseUrl: normalizedBaseUrl,
    request: requestWithPrivateNetworkDefault,
  };
}

export function normalizeOmlxConfiguredCatalogEntry(
  entry: unknown,
): OmlxConfiguredCatalogEntry | null {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const record = entry as Record<string, unknown>;
  if (typeof record.id !== "string" || record.id.trim().length === 0) {
    return null;
  }
  const id = record.id.trim();
  const name = typeof record.name === "string" && record.name.trim().length > 0 ? record.name : id;
  const contextWindow = asPositiveSafeInteger(record.contextWindow);
  const contextTokens = asPositiveSafeInteger(record.contextTokens);
  const reasoning = typeof record.reasoning === "boolean" ? record.reasoning : undefined;
  const input = Array.isArray(record.input)
    ? record.input.filter(
        (item): item is "text" | "image" | "document" =>
          item === "text" || item === "image" || item === "document",
      )
    : undefined;
  const compat =
    record.compat && typeof record.compat === "object" && !Array.isArray(record.compat)
      ? (record.compat as ModelDefinitionConfig["compat"])
      : undefined;
  return {
    id,
    name,
    contextWindow,
    contextTokens,
    reasoning,
    input: input && input.length > 0 ? input : undefined,
    compat,
  };
}

export function normalizeOmlxConfiguredCatalogEntries(
  models: unknown,
): OmlxConfiguredCatalogEntry[] {
  if (!Array.isArray(models)) {
    return [];
  }
  return models
    .map((entry) => normalizeOmlxConfiguredCatalogEntry(entry))
    .filter((entry): entry is OmlxConfiguredCatalogEntry => entry !== null);
}

function friendlyOmlxModelName(entry: Pick<OmlxModelWire, "source_repo_id">, id: string): string {
  const repoId = typeof entry.source_repo_id === "string" ? entry.source_repo_id.trim() : "";
  if (!repoId) {
    return id;
  }
  const slashIndex = repoId.lastIndexOf("/");
  return slashIndex >= 0 ? repoId.slice(slashIndex + 1) : repoId;
}

export function buildOmlxModelName(model: {
  displayName: string;
  vision: boolean;
  reasoning: boolean;
  loaded: boolean;
}): string {
  const tags: string[] = [];
  if (model.vision) {
    tags.push("vision");
  }
  if (model.reasoning) {
    tags.push("reasoning");
  }
  if (model.loaded) {
    tags.push("loaded");
  }
  if (tags.length === 0) {
    return model.displayName;
  }
  return `${model.displayName} (${uniqueStrings(tags).join(", ")})`;
}

/**
 * Base model fields extracted from a single oMLX wire entry.
 * Shared by the setup layer (persists simple names to config) and the runtime
 * discovery path (which enriches the name with vision/reasoning/loaded tags).
 */
export type OmlxModelBase = {
  id: string;
  displayName: string;
  vision: boolean;
  loaded: boolean;
  reasoning: boolean;
  input: Array<"text" | "image">;
  cost: ModelDefinitionConfig["cost"];
  contextWindow: number;
  contextTokens: number;
  maxTokens: number;
};

/**
 * Maps a single oMLX wire entry to its base model fields.
 * Returns null for non-chat entries (embedding/TTS/STT/helper models are wired
 * through the embedding provider or omitted entirely).
 */
export function mapOmlxWireEntry(entry: OmlxModelWire): OmlxModelBase | null {
  if (entry.is_helper === true) {
    return null;
  }
  const modelType = typeof entry.model_type === "string" ? entry.model_type : undefined;
  if (!modelType || !OMLX_CHAT_MODEL_TYPES.has(modelType)) {
    return null;
  }
  const id = typeof entry.id === "string" ? entry.id.trim() : "";
  if (!id) {
    return null;
  }
  const contextWindow = resolveOmlxContextWindow(entry);
  // oMLX reports `max_tokens` as a server-wide generation ceiling, not a per-model
  // output budget: a 32K-context embedding model still advertises 262144, which
  // would yield maxTokens > contextWindow. Bound it by the model's own window and
  // the shared self-hosted output default.
  const advertisedMaxTokens = asPositiveSafeInteger(entry.max_tokens) ?? contextWindow;
  const maxTokens = Math.max(
    1,
    Math.min(contextWindow, advertisedMaxTokens, SELF_HOSTED_DEFAULT_MAX_TOKENS),
  );
  // contextWindow keeps the native maximum; contextTokens is what runtime budgeting
  // and compaction spend. oMLX exposes windows up to 1M that are impractical to fill
  // on local hardware, so cap the spendable budget at OpenClaw's own default ceiling
  // while still shrinking to models that advertise less.
  const contextTokens = Math.min(contextWindow, DEFAULT_CONTEXT_TOKENS);
  const rawDisplayName = friendlyOmlxModelName(entry, id);
  const vision = modelType === "vlm";
  const reasoning = entry.thinking_default === true;
  const loaded = entry.loaded === true;
  return {
    id,
    displayName: rawDisplayName,
    vision,
    loaded,
    reasoning,
    input: vision ? ["text", "image"] : ["text"],
    cost: SELF_HOSTED_DEFAULT_COST,
    contextWindow,
    contextTokens,
    maxTokens,
  };
}

/**
 * Collects non-helper embedding model ids advertised by oMLX discovery.
 * These are deliberately absent from the chat catalog; memory-search selects
 * from this list instead.
 */
export function collectOmlxEmbeddingModelIds(models: readonly OmlxModelWire[]): string[] {
  return models.flatMap((entry) => {
    if (entry.is_helper === true || entry.model_type !== "embedding") {
      return [];
    }
    const id = typeof entry.id === "string" ? entry.id.trim() : "";
    return id ? [id] : [];
  });
}

/**
 * Maps oMLX wire models to config entries using plain display names.
 * Use this for config persistence where runtime state tags are not needed.
 * For runtime discovery with enriched names, use discoverOmlxModels from models.fetch.ts.
 */
export function mapOmlxWireModelsToConfig(models: OmlxModelWire[]): ModelDefinitionConfig[] {
  return models
    .map((entry): ModelDefinitionConfig | null => {
      const base = mapOmlxWireEntry(entry);
      if (!base) {
        return null;
      }
      return {
        id: base.id,
        name: base.displayName,
        reasoning: base.reasoning,
        input: base.input,
        cost: base.cost,
        contextWindow: base.contextWindow,
        contextTokens: base.contextTokens,
        maxTokens: base.maxTokens,
      };
    })
    .filter((entry): entry is ModelDefinitionConfig => entry !== null);
}
