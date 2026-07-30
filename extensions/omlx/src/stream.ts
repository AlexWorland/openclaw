// Omlx plugin module implements stream behavior.
import type { StreamFn } from "openclaw/plugin-sdk/agent-core";
import { streamSimple } from "openclaw/plugin-sdk/llm";
import { createSubsystemLogger } from "openclaw/plugin-sdk/logging-core";
import type { ProviderWrapStreamFnContext } from "openclaw/plugin-sdk/plugin-entry";
import { ssrfPolicyFromHttpBaseUrlAllowedHostname } from "openclaw/plugin-sdk/ssrf-runtime";
import { OMLX_PROVIDER_ID } from "./defaults.js";
import { ensureOmlxModelLoaded } from "./models.fetch.js";
import { normalizeOmlxModelId, resolveOmlxInferenceBase } from "./models.js";
import { resolveOmlxProviderHeaders, resolveOmlxRuntimeApiKey } from "./runtime.js";

const log = createSubsystemLogger("extensions/omlx/stream");

type StreamOptions = Parameters<StreamFn>[2];
type StreamModel = Parameters<StreamFn>[0];

const preloadInFlight = new Map<string, Promise<string | undefined>>();

/**
 * Cooldown state for the oMLX preload endpoint.
 *
 * Without this, every chat request would retry preload ~every request even
 * when oMLX has rejected the load (e.g. unified-memory pressure will keep
 * rejecting until the user frees RAM). The cooldown applies an exponential
 * backoff per preloadKey and, while the cooldown is active, the wrapper skips
 * the preload step entirely and proceeds directly to streaming — the model
 * is often already loaded from oMLX's own tooling, so inference can succeed
 * even when preload keeps being rejected.
 */
type PreloadCooldownEntry = {
  untilMs: number;
  consecutiveFailures: number;
  resolvedModelId?: string;
};

const preloadCooldown = new Map<string, PreloadCooldownEntry>();

const PRELOAD_BACKOFF_BASE_MS = 5_000;
const PRELOAD_BACKOFF_MAX_MS = 300_000;

function computePreloadBackoffMs(consecutiveFailures: number): number {
  const exponent = Math.max(0, consecutiveFailures - 1);
  const raw = PRELOAD_BACKOFF_BASE_MS * 2 ** exponent;
  return Math.min(PRELOAD_BACKOFF_MAX_MS, raw);
}

function recordPreloadSuccess(preloadKey: string): void {
  preloadCooldown.delete(preloadKey);
}

function recordPreloadFailure(
  preloadKey: string,
  now: number,
  resolvedModelId?: string,
): PreloadCooldownEntry {
  const existing = preloadCooldown.get(preloadKey);
  const consecutiveFailures = (existing?.consecutiveFailures ?? 0) + 1;
  const persistedResolvedModelId = resolvedModelId ?? existing?.resolvedModelId;
  const entry: PreloadCooldownEntry = {
    consecutiveFailures,
    untilMs: now + computePreloadBackoffMs(consecutiveFailures),
    ...(persistedResolvedModelId ? { resolvedModelId: persistedResolvedModelId } : {}),
  };
  preloadCooldown.set(preloadKey, entry);
  return entry;
}

function isPreloadCoolingDown(preloadKey: string, now: number): PreloadCooldownEntry | undefined {
  const entry = preloadCooldown.get(preloadKey);
  if (!entry) {
    return undefined;
  }
  if (entry.untilMs <= now) {
    return undefined;
  }
  return entry;
}

function resolveModelHeaders(model: StreamModel): Record<string, string> | undefined {
  if (!model.headers || typeof model.headers !== "object" || Array.isArray(model.headers)) {
    return undefined;
  }
  return model.headers;
}

function toRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}

function shouldPreloadOmlxModels(value: unknown): boolean {
  const providerConfig = toRecord(value);
  const params = toRecord(providerConfig?.params);
  return params?.preload !== false;
}

function withOmlxUsageCompat(model: StreamModel): StreamModel {
  const compat = model.compat && typeof model.compat === "object" ? model.compat : {};
  return {
    ...model,
    compat: { ...compat, supportsUsageInStreaming: true },
  };
}

function withOmlxResolvedModelId(
  model: StreamModel,
  resolvedModelId: string | undefined,
): StreamModel {
  if (!resolvedModelId || model.id === resolvedModelId) {
    return model;
  }
  return {
    ...model,
    id: resolvedModelId,
  };
}

function createPreloadKey(params: { baseUrl: string; modelId: string }) {
  return `${params.baseUrl}::${params.modelId}`;
}

function toOmlxPreloadError(reason: unknown, message: string): Error {
  return reason instanceof Error ? reason : new Error(message, { cause: reason });
}

function waitForOmlxPreload(
  preload: Promise<string | undefined>,
  signal?: AbortSignal,
): Promise<string | undefined> {
  if (!signal) {
    return preload;
  }
  if (signal.aborted) {
    return Promise.reject(toOmlxPreloadError(signal.reason, "oMLX preload aborted"));
  }
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(toOmlxPreloadError(signal.reason, "oMLX preload aborted"));
    signal.addEventListener("abort", onAbort, { once: true });
    void preload.then(
      (modelId) => {
        signal.removeEventListener("abort", onAbort);
        resolve(modelId);
      },
      (error: unknown) => {
        signal.removeEventListener("abort", onAbort);
        reject(toOmlxPreloadError(error, "oMLX model preload failed"));
      },
    );
  });
}

async function ensureOmlxModelLoadedBestEffort(params: {
  baseUrl: string;
  modelId: string;
  options: StreamOptions;
  ctx: ProviderWrapStreamFnContext;
  modelHeaders?: Record<string, string>;
}): Promise<string> {
  const providerConfig = params.ctx.config?.models?.providers?.[OMLX_PROVIDER_ID];
  const providerHeaders = { ...providerConfig?.headers, ...params.modelHeaders };
  const runtimeApiKey =
    typeof params.options?.apiKey === "string" && params.options.apiKey.trim().length > 0
      ? params.options.apiKey.trim()
      : undefined;
  const headers = await resolveOmlxProviderHeaders({
    config: params.ctx.config,
    headers: providerHeaders,
  });
  const configuredApiKey =
    runtimeApiKey !== undefined
      ? undefined
      : await resolveOmlxRuntimeApiKey({
          config: params.ctx.config,
          agentDir: params.ctx.agentDir,
          headers: providerHeaders,
        });

  return await ensureOmlxModelLoaded({
    baseUrl: params.baseUrl,
    apiKey: runtimeApiKey ?? configuredApiKey,
    headers,
    ssrfPolicy: ssrfPolicyFromHttpBaseUrlAllowedHostname(params.baseUrl),
    modelId: params.modelId,
  });
}

export function wrapOmlxInferencePreload(ctx: ProviderWrapStreamFnContext): StreamFn {
  const underlying = ctx.streamFn ?? streamSimple;
  return (model, context, options) => {
    if (model.provider !== OMLX_PROVIDER_ID) {
      return underlying(model, context, options);
    }
    const modelId = normalizeOmlxModelId(model.id);
    if (!modelId) {
      return underlying(model, context, options);
    }
    // Cancellation belongs to this caller; never start or join a shared load after abort.
    options?.signal?.throwIfAborted();
    const providerConfig = ctx.config?.models?.providers?.[OMLX_PROVIDER_ID];
    if (!shouldPreloadOmlxModels(providerConfig)) {
      return underlying(withOmlxUsageCompat(model), context, options);
    }
    const providerBaseUrl = providerConfig?.baseUrl;
    const resolvedBaseUrl = resolveOmlxInferenceBase(
      typeof model.baseUrl === "string" ? model.baseUrl : providerBaseUrl,
    );
    const preloadKey = createPreloadKey({ baseUrl: resolvedBaseUrl, modelId });

    const cooldownEntry = isPreloadCoolingDown(preloadKey, Date.now());
    const existing = preloadInFlight.get(preloadKey);
    const preloadPromise: Promise<string | undefined> | undefined =
      existing ??
      (cooldownEntry
        ? undefined
        : (() => {
            const created = ensureOmlxModelLoadedBestEffort({
              baseUrl: resolvedBaseUrl,
              modelId,
              options,
              ctx,
              modelHeaders: resolveModelHeaders(model),
            })
              .then(
                (resolvedModelId) => {
                  recordPreloadSuccess(preloadKey);
                  return resolvedModelId;
                },
                (error: unknown) => {
                  const entry = recordPreloadFailure(preloadKey, Date.now());
                  throw Object.assign(new Error("preload-failed"), {
                    cause: error,
                    consecutiveFailures: entry.consecutiveFailures,
                    cooldownMs: entry.untilMs - Date.now(),
                  });
                },
              )
              .finally(() => {
                preloadInFlight.delete(preloadKey);
              });
            preloadInFlight.set(preloadKey, created);
            return created;
          })());

    return (async () => {
      let resolvedModelId: string | undefined;
      if (preloadPromise) {
        try {
          resolvedModelId = await waitForOmlxPreload(preloadPromise, options?.signal);
        } catch (error) {
          // A caller owns its wait, not the shared model load needed by other
          // in-flight requests; cancellation must never become preload backoff.
          options?.signal?.throwIfAborted();
          const annotated = error as {
            cause?: unknown;
            consecutiveFailures?: number;
            cooldownMs?: number;
          };
          const cause = annotated.cause ?? error;
          const failures = annotated.consecutiveFailures ?? 1;
          const cooldownSec = Math.max(0, Math.round((annotated.cooldownMs ?? 0) / 1000));
          log.warn(
            `oMLX inference preload failed for "${modelId}" (${failures} consecutive failure${
              failures === 1 ? "" : "s"
            }, next preload attempt skipped for ~${cooldownSec}s); continuing without preload: ${String(cause)}`,
          );
        }
      } else if (cooldownEntry) {
        resolvedModelId = cooldownEntry.resolvedModelId;
        log.debug(
          `oMLX inference preload for "${modelId}" skipped while backoff active (${cooldownEntry.consecutiveFailures} prior failures)`,
        );
      }
      const streamModel = withOmlxResolvedModelId(model, resolvedModelId);
      const stream = underlying(withOmlxUsageCompat(streamModel), context, options);
      const resolvedStream = stream instanceof Promise ? await stream : stream;
      return resolvedStream;
    })();
  };
}
