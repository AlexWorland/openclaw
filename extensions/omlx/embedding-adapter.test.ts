// Omlx tests cover the generic embedding adapter behavior.
import type { MemoryEmbeddingProvider } from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  createOmlxEmbeddingProvider: vi.fn(),
}));

vi.mock("./src/embedding-provider.js", () => ({
  createOmlxEmbeddingProvider: mocks.createOmlxEmbeddingProvider,
  DEFAULT_OMLX_EMBEDDING_MODEL: "Qwen3-Embedding-0.6B-4bit-DWQ",
}));

const { omlxEmbeddingProviderAdapter } = await import("./embedding-adapter.js");

const memoryProvider: MemoryEmbeddingProvider = {
  id: "omlx",
  model: "Qwen3-Embedding-0.6B-4bit-DWQ",
  embedQuery: vi.fn(async () => [1, 0]),
  embedBatch: vi.fn(async (texts) => texts.map(() => [0, 1])),
};

describe("oMLX generic embedding adapter", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.createOmlxEmbeddingProvider.mockResolvedValue({
      provider: memoryProvider,
      client: {
        baseUrl: "http://localhost:8000/v1",
        model: "Qwen3-Embedding-0.6B-4bit-DWQ",
      },
    });
  });

  it("declares the provider id, default model, transport, and auth owner", () => {
    expect(omlxEmbeddingProviderAdapter).toMatchObject({
      id: "omlx",
      defaultModel: "Qwen3-Embedding-0.6B-4bit-DWQ",
      transport: "remote",
      authProviderId: "omlx",
      create: expect.any(Function),
    });
  });

  it("preserves model, dimensions, and runtime identity when creating", async () => {
    const result = await omlxEmbeddingProviderAdapter.create({
      config: {},
      agentDir: "/tmp/openclaw-agent",
      provider: "omlx",
      model: "Qwen3-Embedding-0.6B-4bit-DWQ",
      dimensions: 1024,
    });

    expect(mocks.createOmlxEmbeddingProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        config: {},
        agentDir: "/tmp/openclaw-agent",
        provider: "omlx",
        fallback: "none",
        model: "Qwen3-Embedding-0.6B-4bit-DWQ",
        outputDimensionality: 1024,
      }),
    );
    expect(result.runtime).toEqual({
      id: "omlx",
      cacheKeyData: {
        provider: "omlx",
        baseUrl: "http://localhost:8000/v1",
        model: "Qwen3-Embedding-0.6B-4bit-DWQ",
      },
    });
    expect(result.provider).toMatchObject({ id: "omlx" });
  });

  it("adapts generic query and batch calls without changing text or cancellation", async () => {
    const result = await omlxEmbeddingProviderAdapter.create({
      config: {},
      model: "Qwen3-Embedding-0.6B-4bit-DWQ",
    });
    const provider = result.provider;
    if (!provider) {
      throw new Error("expected an oMLX embedding provider");
    }
    const abortController = new AbortController();

    await expect(
      provider.embed({ text: "query text" }, { signal: abortController.signal }),
    ).resolves.toEqual([1, 0]);
    await expect(
      provider.embedBatch(["document one", { text: "document two" }], {
        signal: abortController.signal,
      }),
    ).resolves.toEqual([
      [0, 1],
      [0, 1],
    ]);

    expect(memoryProvider.embedQuery).toHaveBeenCalledWith("query text", {
      signal: abortController.signal,
    });
    expect(memoryProvider.embedBatch).toHaveBeenCalledWith(["document one", "document two"], {
      signal: abortController.signal,
    });
  });
});
