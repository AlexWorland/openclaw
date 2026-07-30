// Omlx tests cover runtime behavior.
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  resolveApiKeyForProvider: vi.fn(),
}));

vi.mock("openclaw/plugin-sdk/provider-auth-runtime", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  resolveApiKeyForProvider: mocks.resolveApiKeyForProvider,
}));

import {
  buildOmlxAuthHeaders,
  resolveOmlxProviderHeaders,
  resolveOmlxRuntimeApiKey,
} from "./runtime.js";

describe("buildOmlxAuthHeaders", () => {
  it("omits Authorization for the synthetic local placeholder", () => {
    expect(buildOmlxAuthHeaders({ apiKey: "omlx-local" })).toBeUndefined();
  });

  it("sends a Bearer header for a real key and drops any prior Authorization", () => {
    expect(
      buildOmlxAuthHeaders({
        apiKey: "sk-real-key",
        headers: { Authorization: "Bearer stale", "X-Custom": "v" },
      }),
    ).toEqual({ Authorization: "Bearer sk-real-key", "X-Custom": "v" });
  });

  it("adds Content-Type when json is requested", () => {
    expect(buildOmlxAuthHeaders({ json: true })).toEqual({ "Content-Type": "application/json" });
  });

  it("returns undefined when there is nothing to send", () => {
    expect(buildOmlxAuthHeaders({})).toBeUndefined();
  });
});

describe("resolveOmlxProviderHeaders", () => {
  it("returns undefined for non-object header input", async () => {
    await expect(resolveOmlxProviderHeaders({ headers: "not-an-object" })).resolves.toBeUndefined();
  });

  it("sanitizes string headers when no config is provided", async () => {
    await expect(
      resolveOmlxProviderHeaders({ headers: { "X-Custom": "  value  ", Empty: "   " } }),
    ).resolves.toEqual({ "X-Custom": "value" });
  });
});

describe("resolveOmlxRuntimeApiKey", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns undefined when no config is provided", async () => {
    await expect(resolveOmlxRuntimeApiKey({})).resolves.toBeUndefined();
  });

  it("returns the resolved runtime key when available", async () => {
    mocks.resolveApiKeyForProvider.mockResolvedValue({
      apiKey: "sk-real-key",
      source: "profile:x",
    });

    await expect(
      resolveOmlxRuntimeApiKey({
        config: {
          models: {
            providers: {
              omlx: { baseUrl: "http://localhost:8000/v1", api: "openai-completions", models: [] },
            },
          },
        },
      }),
    ).resolves.toBe("sk-real-key");
  });

  it("falls back to the configured apiKey when runtime resolution throws", async () => {
    mocks.resolveApiKeyForProvider.mockRejectedValue(new Error("no profile"));

    await expect(
      resolveOmlxRuntimeApiKey({
        config: {
          models: {
            providers: {
              omlx: {
                baseUrl: "http://localhost:8000/v1",
                api: "openai-completions",
                apiKey: "sk-configured-key",
                models: [],
              },
            },
          },
        },
      }),
    ).resolves.toBe("sk-configured-key");
  });
});
