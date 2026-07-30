// Omlx tests cover models.fetch behavior.
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  fetchWithSsrFGuard: vi.fn(),
}));

vi.mock("openclaw/plugin-sdk/ssrf-runtime", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  fetchWithSsrFGuard: mocks.fetchWithSsrFGuard,
}));

import { discoverOmlxModels, ensureOmlxModelLoaded, fetchOmlxModels } from "./models.fetch.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function okResponse(status = 200): Response {
  return new Response(status === 200 ? "{}" : "load failed", { status });
}

/** oMLX returns the catalog inside a status envelope, not as a bare array. */
function statusResponse(models: unknown[], status = 200): Response {
  return jsonResponse({ final_ceiling: 47, model_count: models.length, models }, status);
}

describe("fetchOmlxModels", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns validated models on success", async () => {
    mocks.fetchWithSsrFGuard.mockResolvedValue({
      response: statusResponse([
        { id: "model-a", model_type: "llm" },
        { id: "model-b", model_type: "vlm" },
      ]),
      release: vi.fn(async () => undefined),
    });

    const result = await fetchOmlxModels({ baseUrl: "http://localhost:8000" });

    expect(result.reachable).toBe(true);
    expect(result.models).toHaveLength(2);
    expect(mocks.fetchWithSsrFGuard).toHaveBeenCalledWith(
      expect.objectContaining({ url: "http://localhost:8000/v1/models/status" }),
    );
  });

  it("skips non-object entries in the discovery array", async () => {
    mocks.fetchWithSsrFGuard.mockResolvedValue({
      response: statusResponse([{ id: "model-a" }, "not-an-object", null, 42]),
      release: vi.fn(async () => undefined),
    });

    const result = await fetchOmlxModels({ baseUrl: "http://localhost:8000" });
    expect(result.models).toHaveLength(1);
  });

  it("stays reachable and records the error when the envelope has no models field", async () => {
    mocks.fetchWithSsrFGuard.mockResolvedValue({
      response: jsonResponse({ error: "unexpected" }),
      release: vi.fn(async () => undefined),
    });

    const result = await fetchOmlxModels({ baseUrl: "http://localhost:8000" });
    expect(result.reachable).toBe(true);
    expect(result.models).toEqual([]);
    expect(String(result.error)).toContain("malformed JSON response");
  });

  it("reports the HTTP status without treating it as unreachable", async () => {
    mocks.fetchWithSsrFGuard.mockResolvedValue({
      response: jsonResponse({ message: "not found" }, 404),
      release: vi.fn(async () => undefined),
    });

    const result = await fetchOmlxModels({ baseUrl: "http://localhost:8000" });
    expect(result.reachable).toBe(true);
    expect(result.status).toBe(404);
    expect(result.models).toEqual([]);
  });

  it("reports unreachable on a transport failure", async () => {
    mocks.fetchWithSsrFGuard.mockRejectedValue(new Error("ECONNREFUSED"));

    const result = await fetchOmlxModels({ baseUrl: "http://localhost:8000" });
    expect(result.reachable).toBe(false);
    expect(result.models).toEqual([]);
  });
});

describe("discoverOmlxModels", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("discovers and maps models, filtering helpers", async () => {
    mocks.fetchWithSsrFGuard.mockResolvedValue({
      response: statusResponse([
        { id: "model-a", model_type: "llm", is_helper: false },
        { id: "drafter", model_type: "llm", is_helper: true },
      ]),
      release: vi.fn(async () => undefined),
    });

    const models = await discoverOmlxModels({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      quiet: true,
    });

    expect(models).toHaveLength(1);
    expect(models[0]?.id).toBe("model-a");
  });

  it("returns an empty list (not a throw) when oMLX is unreachable", async () => {
    mocks.fetchWithSsrFGuard.mockRejectedValue(new Error("ECONNREFUSED"));

    const models = await discoverOmlxModels({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      quiet: true,
    });
    expect(models).toEqual([]);
  });

  it("returns an empty list on a non-2xx discovery response", async () => {
    mocks.fetchWithSsrFGuard.mockResolvedValue({
      response: jsonResponse({}, 500),
      release: vi.fn(async () => undefined),
    });

    const models = await discoverOmlxModels({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      quiet: true,
    });
    expect(models).toEqual([]);
  });
});

describe("ensureOmlxModelLoaded", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("skips the load call when the model is already loaded", async () => {
    mocks.fetchWithSsrFGuard.mockResolvedValueOnce({
      response: statusResponse([{ id: "model-a", loaded: true }]),
      release: vi.fn(async () => undefined),
    });

    await ensureOmlxModelLoaded({ baseUrl: "http://localhost:8000", modelId: "model-a" });

    expect(mocks.fetchWithSsrFGuard).toHaveBeenCalledTimes(1);
  });

  it("POSTs the load endpoint when the model is not loaded", async () => {
    mocks.fetchWithSsrFGuard
      .mockResolvedValueOnce({
        response: statusResponse([{ id: "model-a", loaded: false }]),
        release: vi.fn(async () => undefined),
      })
      .mockResolvedValueOnce({
        response: okResponse(),
        release: vi.fn(async () => undefined),
      });

    const resolved = await ensureOmlxModelLoaded({
      baseUrl: "http://localhost:8000",
      modelId: "model-a",
    });

    expect(resolved).toBe("model-a");
    expect(mocks.fetchWithSsrFGuard).toHaveBeenLastCalledWith(
      expect.objectContaining({
        url: "http://localhost:8000/v1/models/model-a/load",
        init: expect.objectContaining({ method: "POST" }),
      }),
    );
  });

  it("treats an undiscovered model as not loaded and attempts to load it", async () => {
    mocks.fetchWithSsrFGuard
      .mockResolvedValueOnce({
        response: statusResponse([]),
        release: vi.fn(async () => undefined),
      })
      .mockResolvedValueOnce({ response: okResponse(), release: vi.fn(async () => undefined) });

    await ensureOmlxModelLoaded({ baseUrl: "http://localhost:8000", modelId: "unknown-model" });
    expect(mocks.fetchWithSsrFGuard).toHaveBeenCalledTimes(2);
  });

  it("throws when discovery is unreachable", async () => {
    mocks.fetchWithSsrFGuard.mockRejectedValue(new Error("ECONNREFUSED"));

    await expect(
      ensureOmlxModelLoaded({ baseUrl: "http://localhost:8000", modelId: "model-a" }),
    ).rejects.toThrow(/discovery failed/);
  });

  it("throws when the load endpoint returns a non-2xx response", async () => {
    mocks.fetchWithSsrFGuard
      .mockResolvedValueOnce({
        response: statusResponse([{ id: "model-a", loaded: false }]),
        release: vi.fn(async () => undefined),
      })
      .mockResolvedValueOnce({ response: okResponse(500), release: vi.fn(async () => undefined) });

    await expect(
      ensureOmlxModelLoaded({ baseUrl: "http://localhost:8000", modelId: "model-a" }),
    ).rejects.toThrow(/load failed \(500\)/);
  });

  it("rejects an empty model id", async () => {
    await expect(
      ensureOmlxModelLoaded({ baseUrl: "http://localhost:8000", modelId: "   " }),
    ).rejects.toThrow(/model id is required/);
  });
});
