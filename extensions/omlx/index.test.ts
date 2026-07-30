// Omlx tests cover index plugin behavior.
import type { OpenClawConfig, ProviderAuthMethod } from "openclaw/plugin-sdk/plugin-entry";
import { capturePluginRegistration } from "openclaw/plugin-sdk/plugin-test-runtime";
import { CUSTOM_LOCAL_AUTH_MARKER } from "openclaw/plugin-sdk/provider-auth";
import type { ModelProviderConfig } from "openclaw/plugin-sdk/provider-model-shared";
import { beforeEach, describe, expect, it, vi } from "vitest";
import plugin from "./index.js";
import { OMLX_LOCAL_API_KEY_PLACEHOLDER } from "./src/defaults.js";

const fetchOmlxModelsMock = vi.hoisted(() => vi.fn());

vi.mock("./src/models.fetch.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./src/models.fetch.js")>()),
  fetchOmlxModels: fetchOmlxModelsMock,
}));

function registerProvider() {
  const captured = capturePluginRegistration(plugin);
  const provider = captured.providers[0];
  if (!provider) {
    throw new Error("expected the oMLX plugin to register a provider");
  }
  expect(provider.id).toBe("omlx");
  return { provider, captured };
}

function requireOmlxResetValidator(): NonNullable<ProviderAuthMethod["validateNonInteractive"]> {
  const validator = registerProvider().provider.auth[0]?.validateNonInteractive;
  if (!validator) {
    throw new Error("expected the oMLX provider to register a non-interactive reset validator");
  }
  return validator;
}

function createOmlxResetValidationContext(
  opts: Record<string, unknown> = {},
  resolvedApiKey: { key: string; source: "flag" | "env" | "profile" } | null = null,
): Parameters<NonNullable<ProviderAuthMethod["validateNonInteractive"]>>[0] {
  return {
    authChoice: "omlx",
    config: {},
    baseConfig: {},
    opts,
    runtime: {
      log: vi.fn(),
      error: vi.fn(),
      exit: vi.fn() as never,
    },
    resolveApiKey: vi.fn(async () => resolvedApiKey),
  };
}

function createProviderConfig(overrides?: Partial<ModelProviderConfig>): ModelProviderConfig {
  return {
    api: "openai-completions",
    baseUrl: "http://localhost:8000/v1",
    models: [
      {
        id: "Qwen3-8B-4bit",
        name: "Qwen3-8B-4bit",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 32768,
        maxTokens: 8192,
      },
    ],
    ...overrides,
  };
}

describe("omlx plugin", () => {
  beforeEach(() => {
    fetchOmlxModelsMock.mockReset();
  });

  it("registers the oMLX embedding provider", () => {
    const { captured } = registerProvider();
    expect(captured.embeddingProviders.map((provider) => provider.id)).toContain("omlx");
  });

  it("registers a stream wrapper and app-guided setup", () => {
    const { provider } = registerProvider();
    expect(provider.wrapStreamFn).toBeTypeOf("function");
    expect(provider.auth).toHaveLength(1);
    expect(provider.auth[0]?.id).toBe("custom");
    expect(provider.auth[0]?.appGuidedSetup).toMatchObject({
      detect: expect.any(Function),
      prepare: expect.any(Function),
    });
  });

  it("preflights the requested oMLX model before destructive non-interactive reset", async () => {
    fetchOmlxModelsMock.mockResolvedValue({
      reachable: true,
      status: 200,
      models: [{ id: "Qwen3-8B-4bit", model_type: "llm" }],
    });
    const ctx = createOmlxResetValidationContext({
      customBaseUrl: "http://omlx.internal:8000/v1",
      customModelId: "Qwen3-8B-4bit",
    });

    const validateNonInteractive = requireOmlxResetValidator();
    await expect(validateNonInteractive(ctx)).resolves.toBe(true);

    expect(fetchOmlxModelsMock).toHaveBeenCalledWith({
      baseUrl: "http://omlx.internal:8000/v1",
      apiKey: OMLX_LOCAL_API_KEY_PLACEHOLDER,
      timeoutMs: 5000,
    });
    expect(ctx.runtime.exit).not.toHaveBeenCalled();
  });

  it("rejects an unreachable oMLX endpoint before destructive reset", async () => {
    fetchOmlxModelsMock.mockResolvedValue({ reachable: false, models: [] });
    const ctx = createOmlxResetValidationContext({ customBaseUrl: "http://omlx.internal:8000/v1" });

    await expect(requireOmlxResetValidator()(ctx)).resolves.toBe(false);

    expect(ctx.runtime.error).toHaveBeenCalledWith(
      "oMLX could not be reached at http://omlx.internal:8000/v1.\nStart the oMLX server and re-run setup.",
    );
    expect(ctx.runtime.exit).toHaveBeenCalledWith(1);
  });

  it("rejects a missing requested oMLX model before destructive reset", async () => {
    fetchOmlxModelsMock.mockResolvedValue({
      reachable: true,
      status: 200,
      models: [{ id: "other-model", model_type: "llm" }],
    });
    const ctx = createOmlxResetValidationContext({
      customBaseUrl: "http://omlx.internal:8000/v1",
      customModelId: "Qwen3-8B-4bit",
    });

    await expect(requireOmlxResetValidator()(ctx)).resolves.toBe(false);
    expect(ctx.runtime.error).toHaveBeenCalledWith(
      "oMLX model Qwen3-8B-4bit was not found at http://omlx.internal:8000/v1.\nAvailable models: other-model",
    );
    expect(ctx.runtime.exit).toHaveBeenCalledWith(1);
  });

  it("rejects an oMLX endpoint without a usable LLM/VLM model before destructive reset", async () => {
    fetchOmlxModelsMock.mockResolvedValue({
      reachable: true,
      status: 200,
      models: [{ id: "embed-model", model_type: "embedding" }],
    });
    const ctx = createOmlxResetValidationContext({ customBaseUrl: "http://omlx.internal:8000/v1" });

    await expect(requireOmlxResetValidator()(ctx)).resolves.toBe(false);
    expect(ctx.runtime.error).toHaveBeenCalledWith(
      "No oMLX LLM/VLM models were found at http://omlx.internal:8000/v1.\nLoad at least one model in oMLX, then re-run setup.",
    );
    expect(ctx.runtime.exit).toHaveBeenCalledWith(1);
  });

  it("canonicalizes base URLs during provider normalization", () => {
    const { provider } = registerProvider();
    const providerConfig = createProviderConfig({ baseUrl: "http://localhost:8000" });

    expect(provider.normalizeConfig?.({ provider: "omlx", providerConfig })).toEqual({
      ...providerConfig,
      baseUrl: "http://localhost:8000/v1",
      request: { allowPrivateNetwork: true },
    });
  });

  it("synthesizes placeholder auth for configured omlx models without an API key", () => {
    const { provider } = registerProvider();

    expect(
      provider.resolveSyntheticAuth?.({
        provider: "omlx",
        config: {} as OpenClawConfig,
        providerConfig: createProviderConfig(),
      }),
    ).toEqual({
      apiKey: CUSTOM_LOCAL_AUTH_MARKER,
      source: "models.providers.omlx (synthetic local key)",
      mode: "api-key",
    });
  });

  it("does not synthesize placeholder auth when Authorization header is configured", () => {
    const { provider } = registerProvider();

    expect(
      provider.resolveSyntheticAuth?.({
        provider: "omlx",
        config: {} as OpenClawConfig,
        providerConfig: createProviderConfig({
          headers: { Authorization: "Bearer proxy-token" },
        }),
      }),
    ).toBeUndefined();
  });

  it("defers stored omlx-local profile auth so real credentials can win", () => {
    const { provider } = registerProvider();

    expect(
      provider.shouldDeferSyntheticProfileAuth?.({
        provider: "omlx",
        config: {} as OpenClawConfig,
        providerConfig: createProviderConfig(),
        resolvedApiKey: OMLX_LOCAL_API_KEY_PLACEHOLDER,
      }),
    ).toBe(true);

    expect(
      provider.shouldDeferSyntheticProfileAuth?.({
        provider: "omlx",
        config: {} as OpenClawConfig,
        providerConfig: createProviderConfig(),
        resolvedApiKey: CUSTOM_LOCAL_AUTH_MARKER,
      }),
    ).toBe(true);

    expect(
      provider.shouldDeferSyntheticProfileAuth?.({
        provider: "omlx",
        config: {} as OpenClawConfig,
        providerConfig: createProviderConfig(),
        resolvedApiKey: "omlx-real-key",
      }),
    ).toBe(false);
  });

  it("augments the catalog with configured omlx models", () => {
    const { provider } = registerProvider();
    const config = {
      models: {
        providers: {
          omlx: {
            models: [
              {
                id: "Qwen3-8B-4bit",
                name: "Qwen3 8B",
                contextWindow: 32768,
                reasoning: true,
                input: ["text", "image"],
              },
              { id: "phi-4" },
              { id: " ", name: "ignored" },
            ],
          },
        },
      },
    } as unknown as OpenClawConfig;

    expect(
      provider.augmentModelCatalog?.({ config, agentDir: "/tmp/openclaw", env: {}, entries: [] }),
    ).toEqual([
      {
        provider: "omlx",
        id: "Qwen3-8B-4bit",
        name: "Qwen3 8B",
        compat: { supportsUsageInStreaming: true },
        contextWindow: 32768,
        contextTokens: undefined,
        reasoning: true,
        input: ["text", "image"],
      },
      {
        provider: "omlx",
        id: "phi-4",
        name: "phi-4",
        compat: { supportsUsageInStreaming: true },
        contextWindow: undefined,
        contextTokens: undefined,
        reasoning: undefined,
        input: undefined,
      },
    ]);
  });
});
