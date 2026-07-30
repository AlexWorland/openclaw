// Omlx tests cover embedding-provider behavior.
import type { MemoryEmbeddingProviderCreateOptions } from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  createRemoteEmbeddingProvider: vi.fn(),
  ensureOmlxModelLoaded: vi.fn(),
  resolveOmlxProviderHeaders: vi.fn(),
  resolveOmlxRuntimeApiKey: vi.fn(),
}));

vi.mock("openclaw/plugin-sdk/memory-core-host-engine-embeddings", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  createRemoteEmbeddingProvider: mocks.createRemoteEmbeddingProvider,
}));

vi.mock("./models.fetch.js", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureOmlxModelLoaded: mocks.ensureOmlxModelLoaded,
}));

vi.mock("./runtime.js", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  resolveOmlxProviderHeaders: mocks.resolveOmlxProviderHeaders,
  resolveOmlxRuntimeApiKey: mocks.resolveOmlxRuntimeApiKey,
}));

const { createOmlxEmbeddingProvider } = await import("./embedding-provider.js");

const remoteProviderStub = {
  id: "omlx",
  model: "Qwen3-Embedding-0.6B-4bit-DWQ",
  embedQuery: vi.fn(async () => [1, 0]),
  embedBatch: vi.fn(async (texts: string[]) => texts.map(() => [0, 1])),
};

function buildOptions(
  overrides: Partial<MemoryEmbeddingProviderCreateOptions> = {},
): MemoryEmbeddingProviderCreateOptions {
  return {
    config: {},
    provider: "omlx",
    model: "Qwen3-Embedding-0.6B-4bit-DWQ",
    ...overrides,
  } as MemoryEmbeddingProviderCreateOptions;
}

describe("createOmlxEmbeddingProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.createRemoteEmbeddingProvider.mockReturnValue(remoteProviderStub);
    mocks.ensureOmlxModelLoaded.mockResolvedValue(undefined);
    mocks.resolveOmlxProviderHeaders.mockResolvedValue(undefined);
    mocks.resolveOmlxRuntimeApiKey.mockResolvedValue(undefined);
  });

  it("resolves the default local base URL and model", async () => {
    const { client } = await createOmlxEmbeddingProvider(buildOptions());

    expect(client.baseUrl).toBe("http://localhost:8000/v1");
    expect(client.model).toBe("Qwen3-Embedding-0.6B-4bit-DWQ");
  });

  it("preloads the target model before returning the provider", async () => {
    await createOmlxEmbeddingProvider(buildOptions());

    expect(mocks.ensureOmlxModelLoaded).toHaveBeenCalledWith(
      expect.objectContaining({
        baseUrl: "http://localhost:8000/v1",
        modelId: "Qwen3-Embedding-0.6B-4bit-DWQ",
      }),
    );
  });

  it("continues (best-effort) when the warmup preload fails", async () => {
    mocks.ensureOmlxModelLoaded.mockRejectedValue(new Error("oMLX unreachable"));

    await expect(createOmlxEmbeddingProvider(buildOptions())).resolves.toBeDefined();
  });

  it("uses the configured provider base URL when set", async () => {
    const { client } = await createOmlxEmbeddingProvider(
      buildOptions({
        config: {
          models: {
            providers: {
              omlx: {
                baseUrl: "http://192.168.1.50:8000/v1",
                api: "openai-completions",
                models: [],
              },
            },
          },
        },
      }),
    );

    expect(client.baseUrl).toBe("http://192.168.1.50:8000/v1");
  });

  it("strips the omlx/ prefix from the requested model", async () => {
    const { client } = await createOmlxEmbeddingProvider(
      buildOptions({ model: "omlx/custom-embed-model" }),
    );
    expect(client.model).toBe("custom-embed-model");
  });

  it("delegates embedQuery/embedBatch to the underlying remote provider", async () => {
    const { provider } = await createOmlxEmbeddingProvider(buildOptions());

    await expect(provider.embedQuery("hello")).resolves.toEqual([1, 0]);
    await expect(provider.embedBatch(["a", "b"])).resolves.toEqual([
      [0, 1],
      [0, 1],
    ]);
  });
});
