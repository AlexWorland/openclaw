// Omlx tests cover stream behavior.
import type { StreamFn } from "openclaw/plugin-sdk/agent-core";
import type { Context, Model } from "openclaw/plugin-sdk/llm";
import type { ProviderWrapStreamFnContext } from "openclaw/plugin-sdk/plugin-entry";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  ensureOmlxModelLoaded: vi.fn(),
  resolveOmlxProviderHeaders: vi.fn(),
  resolveOmlxRuntimeApiKey: vi.fn(),
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

const { wrapOmlxInferencePreload } = await import("./stream.js");

describe("wrapOmlxInferencePreload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.ensureOmlxModelLoaded.mockResolvedValue("model-a");
    mocks.resolveOmlxProviderHeaders.mockResolvedValue(undefined);
    mocks.resolveOmlxRuntimeApiKey.mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function buildCtx(streamFn: StreamFn): ProviderWrapStreamFnContext {
    return { streamFn, config: {} } as unknown as ProviderWrapStreamFnContext;
  }

  it("passes non-oMLX models straight through without preloading", async () => {
    const underlying: StreamFn = vi.fn(() => ({}) as ReturnType<StreamFn>);
    const wrapped = wrapOmlxInferencePreload(buildCtx(underlying));

    const model = { provider: "openai", id: "gpt-5.4" } as unknown as Model<"openai-completions">;
    await wrapped(model, { messages: [] } as Context, {});

    expect(underlying).toHaveBeenCalledOnce();
    expect(mocks.ensureOmlxModelLoaded).not.toHaveBeenCalled();
  });

  it("loads the model before invoking the underlying stream for oMLX models", async () => {
    const underlying: StreamFn = vi.fn(() => ({}) as ReturnType<StreamFn>);
    const wrapped = wrapOmlxInferencePreload(buildCtx(underlying));

    const model = {
      provider: "omlx",
      id: "model-a",
      baseUrl: "http://localhost:8000/v1",
    } as unknown as Model<"openai-completions">;
    await wrapped(model, { messages: [] } as Context, {});

    expect(mocks.ensureOmlxModelLoaded).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://localhost:8000/v1", modelId: "model-a" }),
    );
    expect(underlying).toHaveBeenCalledOnce();
  });

  it("forces supportsUsageInStreaming compat on the model passed to the underlying stream", async () => {
    let capturedModel: Model<"openai-completions"> | undefined;
    const underlying: StreamFn = vi.fn((model) => {
      capturedModel = model as Model<"openai-completions">;
      return {} as ReturnType<StreamFn>;
    });
    const wrapped = wrapOmlxInferencePreload(buildCtx(underlying));

    const model = {
      provider: "omlx",
      id: "model-compat",
      baseUrl: "http://localhost:8000/v1",
    } as unknown as Model<"openai-completions">;
    await wrapped(model, { messages: [] } as Context, {});

    expect(capturedModel?.compat).toMatchObject({ supportsUsageInStreaming: true });
  });

  it("skips preload when the provider config sets params.preload: false", async () => {
    const underlying: StreamFn = vi.fn(() => ({}) as ReturnType<StreamFn>);
    const ctx = {
      streamFn: underlying,
      config: { models: { providers: { omlx: { params: { preload: false } } } } },
    } as unknown as ProviderWrapStreamFnContext;
    const wrapped = wrapOmlxInferencePreload(ctx);

    const model = {
      provider: "omlx",
      id: "model-no-preload",
      baseUrl: "http://localhost:8000/v1",
    } as unknown as Model<"openai-completions">;
    await wrapped(model, { messages: [] } as Context, {});

    expect(mocks.ensureOmlxModelLoaded).not.toHaveBeenCalled();
    expect(underlying).toHaveBeenCalledOnce();
  });

  it("continues to inference (best-effort) when preload fails", async () => {
    mocks.ensureOmlxModelLoaded.mockRejectedValue(new Error("ECONNREFUSED"));
    const underlying: StreamFn = vi.fn(() => ({}) as ReturnType<StreamFn>);
    const wrapped = wrapOmlxInferencePreload(buildCtx(underlying));

    const model = {
      provider: "omlx",
      id: "model-preload-fails",
      baseUrl: "http://localhost:8000/v1",
    } as unknown as Model<"openai-completions">;
    await expect(wrapped(model, { messages: [] } as Context, {})).resolves.toBeDefined();

    expect(underlying).toHaveBeenCalledOnce();
  });

  it("dedupes concurrent preload requests for the same model", async () => {
    const underlying: StreamFn = vi.fn(() => ({}) as ReturnType<StreamFn>);
    const wrapped = wrapOmlxInferencePreload(buildCtx(underlying));

    const model = {
      provider: "omlx",
      id: "model-dedupe",
      baseUrl: "http://localhost:8000/v1",
    } as unknown as Model<"openai-completions">;
    await Promise.all([
      wrapped(model, { messages: [] } as Context, {}),
      wrapped(model, { messages: [] } as Context, {}),
    ]);

    expect(mocks.ensureOmlxModelLoaded).toHaveBeenCalledOnce();
  });

  it("throws synchronously on an already-aborted signal instead of preloading", () => {
    const controller = new AbortController();
    controller.abort(new Error("cancelled"));
    const underlying: StreamFn = vi.fn(() => ({}) as ReturnType<StreamFn>);
    const wrapped = wrapOmlxInferencePreload(buildCtx(underlying));

    const model = {
      provider: "omlx",
      id: "model-aborted",
      baseUrl: "http://localhost:8000/v1",
    } as unknown as Model<"openai-completions">;
    expect(() =>
      wrapped(model, { messages: [] } as Context, { signal: controller.signal }),
    ).toThrow();
    expect(underlying).not.toHaveBeenCalled();
  });
});
