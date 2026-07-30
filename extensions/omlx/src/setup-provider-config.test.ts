// Omlx tests cover setup provider-config behavior.
import type { OpenClawConfig } from "openclaw/plugin-sdk/provider-auth";
import type { ModelDefinitionConfig } from "openclaw/plugin-sdk/provider-model-shared";
import { describe, expect, it } from "vitest";
import { OMLX_DEFAULT_API_KEY_ENV_VAR, OMLX_LOCAL_API_KEY_PLACEHOLDER } from "./defaults.js";
import {
  buildOmlxSetupProviderConfig,
  mergeDiscoveredModels,
  resolvePersistedOmlxApiKey,
  stripOmlxStoredAuthConfig,
} from "./setup-provider-config.js";

function createModel(id: string): ModelDefinitionConfig {
  return {
    id,
    name: id,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32768,
    maxTokens: 8192,
  };
}

describe("stripOmlxStoredAuthConfig", () => {
  it("removes only omlx profiles and order entries", () => {
    const cfg = {
      auth: {
        profiles: {
          "omlx:default": { provider: "omlx" },
          "openai:default": { provider: "openai" },
        },
        order: { omlx: ["omlx:default"], openai: ["openai:default"] },
      },
    } as unknown as OpenClawConfig;

    const result = stripOmlxStoredAuthConfig(cfg);

    expect(result.auth?.profiles).toEqual({ "openai:default": { provider: "openai" } });
    expect(result.auth?.order).toEqual({ openai: ["openai:default"] });
  });

  it("drops the auth block entirely when omlx was its only content", () => {
    const cfg = {
      auth: { profiles: { "omlx:default": { provider: "omlx" } }, order: { omlx: ["x"] } },
    } as unknown as OpenClawConfig;

    expect(stripOmlxStoredAuthConfig(cfg).auth).toBeUndefined();
  });
});

describe("buildOmlxSetupProviderConfig", () => {
  it("defaults api to openai-completions and persists the given base URL and models", () => {
    const result = buildOmlxSetupProviderConfig({
      existingProvider: undefined,
      baseUrl: "http://localhost:8000/v1",
      headers: undefined,
      models: [createModel("model-a")],
    });

    expect(result).toMatchObject({
      baseUrl: "http://localhost:8000/v1",
      api: "openai-completions",
      models: [createModel("model-a")],
    });
  });

  it("omits auth when the persisted key is the non-secret local placeholder", () => {
    const result = buildOmlxSetupProviderConfig({
      existingProvider: undefined,
      baseUrl: "http://localhost:8000/v1",
      apiKey: OMLX_LOCAL_API_KEY_PLACEHOLDER,
      headers: undefined,
      models: [],
    });

    expect(result.auth).toBeUndefined();
    expect(result.apiKey).toBe(OMLX_LOCAL_API_KEY_PLACEHOLDER);
  });

  it("sets api-key auth for a real credential", () => {
    const result = buildOmlxSetupProviderConfig({
      existingProvider: undefined,
      baseUrl: "http://localhost:8000/v1",
      apiKey: "sk-real-key",
      headers: undefined,
      models: [],
    });

    expect(result.auth).toBe("api-key");
  });

  it("preserves unrelated existing provider fields while replacing auth fields", () => {
    const result = buildOmlxSetupProviderConfig({
      existingProvider: {
        baseUrl: "http://stale:8000/v1",
        api: "openai-completions",
        timeoutSeconds: 300,
        apiKey: "stale-key",
        auth: "api-key",
        models: [],
      },
      baseUrl: "http://localhost:8000/v1",
      headers: undefined,
      models: [],
    });

    expect(result.timeoutSeconds).toBe(300);
    expect(result.baseUrl).toBe("http://localhost:8000/v1");
    expect(result.apiKey).toBeUndefined();
    expect(result.auth).toBeUndefined();
  });
});

describe("resolvePersistedOmlxApiKey", () => {
  it("prefers the fallback key when explicit api-key auth requests it", () => {
    expect(
      resolvePersistedOmlxApiKey({
        currentApiKey: "stale-key",
        explicitAuth: "api-key",
        fallbackApiKey: OMLX_DEFAULT_API_KEY_ENV_VAR,
        preferFallbackApiKey: true,
        hasModels: true,
      }),
    ).toBe(OMLX_DEFAULT_API_KEY_ENV_VAR);
  });

  it("keeps a real current key under api-key auth when no fallback is preferred", () => {
    expect(
      resolvePersistedOmlxApiKey({
        currentApiKey: "sk-real-key",
        explicitAuth: "api-key",
        fallbackApiKey: OMLX_DEFAULT_API_KEY_ENV_VAR,
        hasModels: true,
      }),
    ).toBe("sk-real-key");
  });

  it("persists the local placeholder for a keyless local server with models", () => {
    expect(
      resolvePersistedOmlxApiKey({
        currentApiKey: undefined,
        explicitAuth: undefined,
        fallbackApiKey: undefined,
        hasModels: true,
      }),
    ).toBe(OMLX_LOCAL_API_KEY_PLACEHOLDER);
  });

  it("persists nothing when an Authorization header already carries auth", () => {
    expect(
      resolvePersistedOmlxApiKey({
        currentApiKey: undefined,
        explicitAuth: undefined,
        fallbackApiKey: undefined,
        hasModels: true,
        hasAuthorizationHeader: true,
      }),
    ).toBeUndefined();
  });

  it("persists nothing before any model is known", () => {
    expect(
      resolvePersistedOmlxApiKey({
        currentApiKey: undefined,
        explicitAuth: undefined,
        fallbackApiKey: undefined,
        hasModels: false,
      }),
    ).toBeUndefined();
  });
});

describe("mergeDiscoveredModels", () => {
  it("keeps explicit entries first and appends unique discovered entries", () => {
    expect(
      mergeDiscoveredModels({
        explicitModels: [createModel("explicit-a")],
        discoveredModels: [createModel("explicit-a"), createModel("discovered-b")],
      }).map((model) => model.id),
    ).toEqual(["explicit-a", "discovered-b"]);
  });

  it("returns the other list when either side is empty or missing", () => {
    expect(mergeDiscoveredModels({ discoveredModels: [createModel("a")] })).toHaveLength(1);
    expect(mergeDiscoveredModels({ explicitModels: [createModel("a")] })).toHaveLength(1);
    expect(mergeDiscoveredModels({})).toEqual([]);
  });
});
