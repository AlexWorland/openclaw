// Omlx tests cover setup plugin behavior.
import type { OpenClawConfig } from "openclaw/plugin-sdk/provider-auth";
import type { ModelDefinitionConfig } from "openclaw/plugin-sdk/provider-model-shared";
import type {
  ProviderAuthMethodNonInteractiveContext,
  ProviderCatalogContext,
  ProviderPrepareDynamicModelContext,
} from "openclaw/plugin-sdk/provider-setup";
import type { WizardPrompter } from "openclaw/plugin-sdk/setup";
import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";
import { OMLX_DEFAULT_INFERENCE_BASE_URL } from "./defaults.js";

const fetchOmlxModelsMock = vi.hoisted(() => vi.fn());
const discoverOmlxModelsMock = vi.hoisted(() => vi.fn());
const configureSelfHostedNonInteractiveMock = vi.hoisted(() => vi.fn());
const removeProviderAuthProfilesWithLockMock = vi.hoisted(() => vi.fn());
const resolveOmlxRequestContextMock = vi.hoisted(() => vi.fn());

vi.mock("./models.fetch.js", () => ({
  fetchOmlxModels: (...args: unknown[]) => fetchOmlxModelsMock(...args),
  discoverOmlxModels: (...args: unknown[]) => discoverOmlxModelsMock(...args),
  ensureOmlxModelLoaded: vi.fn(),
}));

vi.mock("./runtime.js", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  resolveOmlxRequestContext: (...args: unknown[]) => resolveOmlxRequestContextMock(...args),
}));

vi.mock("openclaw/plugin-sdk/provider-auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("openclaw/plugin-sdk/provider-auth")>();
  return {
    ...actual,
    removeProviderAuthProfilesWithLock: (...args: unknown[]) =>
      removeProviderAuthProfilesWithLockMock(...args),
  };
});

vi.mock("openclaw/plugin-sdk/provider-setup", async (importOriginal) => {
  const actual = await importOriginal<typeof import("openclaw/plugin-sdk/provider-setup")>();
  return {
    ...actual,
    configureOpenAICompatibleSelfHostedProviderNonInteractive: (...args: unknown[]) =>
      configureSelfHostedNonInteractiveMock(...args),
  };
});

afterAll(() => {
  vi.doUnmock("./models.fetch.js");
  vi.doUnmock("./runtime.js");
  vi.doUnmock("openclaw/plugin-sdk/provider-auth");
  vi.doUnmock("openclaw/plugin-sdk/provider-setup");
  vi.resetModules();
});

const {
  configureOmlxNonInteractive,
  discoverOmlxProvider,
  prepareAppGuidedOmlxSetup,
  prepareOmlxDynamicModels,
  promptAndConfigureOmlxInteractive,
} = await import("./setup.js");

function createModel(id: string, name = id): ModelDefinitionConfig {
  return {
    id,
    name,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32768,
    maxTokens: 8192,
  };
}

function createWireModel(overrides: Record<string, unknown> = {}) {
  return {
    id: "Qwen3-8B-4bit",
    model_type: "llm",
    max_context_window: 32768,
    thinking_default: false,
    is_helper: false,
    loaded: false,
    ...overrides,
  };
}

function buildDiscoveryContext(params?: {
  config?: OpenClawConfig;
  apiKey?: string;
  discoveryApiKey?: string;
  env?: NodeJS.ProcessEnv;
}): ProviderCatalogContext {
  return {
    config: params?.config ?? ({} as OpenClawConfig),
    env: params?.env ?? {},
    resolveProviderApiKey: () => ({
      apiKey: params?.apiKey,
      discoveryApiKey: params?.discoveryApiKey,
    }),
  } as unknown as ProviderCatalogContext;
}

function createQueuedWizardPrompterHarness(textValues: string[]): {
  prompter: WizardPrompter;
  note: ReturnType<typeof vi.fn>;
} {
  const queue = [...textValues];
  const note = vi.fn(async () => undefined);
  const prompter: WizardPrompter = {
    intro: async () => {},
    outro: async () => {},
    note,
    select: async <T>(params: { options: Array<{ value: T }> }) => {
      const firstOption = params.options[0];
      if (!firstOption) {
        throw new Error("select called without options");
      }
      return firstOption.value;
    },
    multiselect: async () => [],
    text: vi.fn(async () => queue.shift() ?? ""),
    confirm: async () => false,
    progress: () => ({ update: () => {}, stop: () => {} }),
  };
  return { prompter, note };
}

function buildNonInteractiveContext(
  params: Partial<ProviderAuthMethodNonInteractiveContext["opts"]> & {
    config?: OpenClawConfig;
    resolvedApiKey?: { key: string; source: "flag" | "env" | "profile" } | null;
  } = {},
): ProviderAuthMethodNonInteractiveContext & {
  runtime: { error: ReturnType<typeof vi.fn>; exit: ReturnType<typeof vi.fn> };
} {
  const error = vi.fn();
  const exit = vi.fn();
  const { config, resolvedApiKey, ...opts } = params;
  return {
    authChoice: "omlx",
    config: config ?? ({ models: { providers: {} } } as OpenClawConfig),
    baseConfig: config ?? ({} as OpenClawConfig),
    opts,
    runtime: { error, exit, log: vi.fn() },
    resolveApiKey: vi.fn(async () => resolvedApiKey ?? { key: "sk-test-key", source: "flag" }),
    toApiKeyCredential: vi.fn(),
  } as unknown as ProviderAuthMethodNonInteractiveContext & {
    runtime: { error: ReturnType<typeof vi.fn>; exit: ReturnType<typeof vi.fn> };
  };
}

describe("discoverOmlxProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("passes through explicit models without live discovery", async () => {
    const explicitModels = [createModel("Qwen3-8B-4bit")];
    const ctx = buildDiscoveryContext({
      config: {
        models: {
          providers: {
            omlx: {
              baseUrl: "http://localhost:8000/v1",
              api: "openai-completions",
              models: explicitModels,
            },
          },
        },
      },
    });

    const result = await discoverOmlxProvider(ctx);

    expect(fetchOmlxModelsMock).not.toHaveBeenCalled();
    expect(result?.provider.models).toEqual(explicitModels);
  });

  it("merges live-discovered models when no explicit models are configured", async () => {
    discoverOmlxModelsMock.mockResolvedValue([createModel("model-discovered")]);
    const ctx = buildDiscoveryContext({ apiKey: "sk-real-key" });

    const result = await discoverOmlxProvider(ctx);

    expect(result?.provider.models).toEqual([createModel("model-discovered")]);
    expect(result?.provider.baseUrl).toBe(OMLX_DEFAULT_INFERENCE_BASE_URL);
  });

  it("returns null when discovery finds nothing and no auth is configured", async () => {
    discoverOmlxModelsMock.mockResolvedValue([]);
    const ctx = buildDiscoveryContext();

    await expect(discoverOmlxProvider(ctx)).resolves.toBeNull();
  });
});

describe("prepareAppGuidedOmlxSetup", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("selects a qualifying model and proposes a config patch", async () => {
    fetchOmlxModelsMock.mockResolvedValue({
      reachable: true,
      status: 200,
      models: [createWireModel()],
    });

    const result = await prepareAppGuidedOmlxSetup({
      config: {} as OpenClawConfig,
      env: {},
    });

    expect(result?.defaultModel).toBe("omlx/Qwen3-8B-4bit");
  });

  it("returns null when no model meets the minimum context window", async () => {
    fetchOmlxModelsMock.mockResolvedValue({
      reachable: true,
      status: 200,
      models: [createWireModel({ max_context_window: 2048 })],
    });

    await expect(
      prepareAppGuidedOmlxSetup({ config: {} as OpenClawConfig, env: {} }),
    ).resolves.toBeNull();
  });

  it("returns null when oMLX is unreachable", async () => {
    fetchOmlxModelsMock.mockResolvedValue({ reachable: false, models: [], error: new Error("x") });

    await expect(
      prepareAppGuidedOmlxSetup({ config: {} as OpenClawConfig, env: {} }),
    ).resolves.toBeNull();
  });

  it("excludes helper (draft) models from app-guided selection", async () => {
    fetchOmlxModelsMock.mockResolvedValue({
      reachable: true,
      status: 200,
      models: [createWireModel({ is_helper: true })],
    });

    await expect(
      prepareAppGuidedOmlxSetup({ config: {} as OpenClawConfig, env: {} }),
    ).resolves.toBeNull();
  });
});

describe("promptAndConfigureOmlxInteractive", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("configures the provider from prompted base URL and discovered models", async () => {
    fetchOmlxModelsMock.mockResolvedValue({
      reachable: true,
      status: 200,
      models: [createWireModel()],
    });
    const { prompter } = createQueuedWizardPrompterHarness(["http://localhost:8000", ""]);

    const result = await promptAndConfigureOmlxInteractive({
      config: {} as OpenClawConfig,
      prompter,
    });

    expect(result.defaultModel).toBe("omlx/Qwen3-8B-4bit");
    expect(result.configPatch?.models?.providers?.omlx?.baseUrl).toBe("http://localhost:8000/v1");
  });

  it("throws a cancellation error when oMLX is unreachable at the prompted base URL", async () => {
    fetchOmlxModelsMock.mockResolvedValue({ reachable: false, models: [], error: new Error("x") });
    const { prompter, note } = createQueuedWizardPrompterHarness(["http://localhost:8000", ""]);

    await expect(
      promptAndConfigureOmlxInteractive({ config: {} as OpenClawConfig, prompter }),
    ).rejects.toThrow();
    expect(note).toHaveBeenCalled();
  });
});

describe("configureOmlxNonInteractive", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("delegates to the shared self-hosted helper and post-patches discovered models", async () => {
    fetchOmlxModelsMock.mockResolvedValue({
      reachable: true,
      status: 200,
      models: [createWireModel()],
    });
    configureSelfHostedNonInteractiveMock.mockResolvedValue({
      models: {
        providers: {
          omlx: {
            baseUrl: "http://localhost:8000/v1",
            api: "openai-completions",
            models: [],
          },
        },
      },
    });
    const ctx = buildNonInteractiveContext({
      customModelId: "Qwen3-8B-4bit",
      resolvedApiKey: { key: "sk-real-key", source: "flag" },
    });

    const configured = await configureOmlxNonInteractive(ctx);

    expect(configured?.models?.providers?.omlx?.models).toHaveLength(1);
    expect(configureSelfHostedNonInteractiveMock).toHaveBeenCalled();
  });

  it("errors and exits when the requested model is not available", async () => {
    fetchOmlxModelsMock.mockResolvedValue({
      reachable: true,
      status: 200,
      models: [createWireModel({ id: "other-model" })],
    });
    const ctx = buildNonInteractiveContext({
      customModelId: "Qwen3-8B-4bit",
      resolvedApiKey: { key: "sk-real-key", source: "flag" },
    });

    await expect(configureOmlxNonInteractive(ctx)).resolves.toBeNull();
    expect(ctx.runtime.error).toHaveBeenCalled();
    expect(ctx.runtime.exit).toHaveBeenCalledWith(1);
  });

  it("errors and exits when oMLX is unreachable", async () => {
    fetchOmlxModelsMock.mockResolvedValue({ reachable: false, models: [], error: new Error("x") });
    const ctx = buildNonInteractiveContext({
      resolvedApiKey: { key: "sk-real-key", source: "flag" },
    });

    await expect(configureOmlxNonInteractive(ctx)).resolves.toBeNull();
    expect(ctx.runtime.exit).toHaveBeenCalledWith(1);
  });
});

describe("prepareOmlxDynamicModels", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("overlays provider/api/baseUrl onto discovered models", async () => {
    discoverOmlxModelsMock.mockResolvedValue([createModel("model-a")]);
    resolveOmlxRequestContextMock.mockResolvedValue({ apiKey: undefined, headers: undefined });
    const ctx = {
      config: {} as OpenClawConfig,
      providerConfig: {
        baseUrl: "http://localhost:8000/v1",
        api: "openai-completions",
        models: [],
      },
    } as unknown as ProviderPrepareDynamicModelContext;

    const models = await prepareOmlxDynamicModels(ctx);

    expect(models).toEqual([
      expect.objectContaining({
        id: "model-a",
        provider: "omlx",
        api: "openai-completions",
        baseUrl: "http://localhost:8000/v1",
      }),
    ]);
  });
});
