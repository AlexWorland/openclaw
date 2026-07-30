// Omlx tests cover provider-auth behavior.
import { describe, expect, it } from "vitest";
import {
  hasOmlxAuthorizationHeader,
  resolveOmlxProviderAuthMode,
  shouldUseOmlxApiKeyPlaceholder,
  shouldUseOmlxSyntheticAuth,
} from "./provider-auth.js";

describe("hasOmlxAuthorizationHeader", () => {
  it("detects a configured Authorization header case-insensitively", () => {
    expect(hasOmlxAuthorizationHeader({ authorization: "Bearer x" })).toBe(true);
    expect(hasOmlxAuthorizationHeader({ Authorization: "Bearer x" })).toBe(true);
  });

  it("returns false for missing, non-object, or empty headers", () => {
    expect(hasOmlxAuthorizationHeader(undefined)).toBe(false);
    expect(hasOmlxAuthorizationHeader([])).toBe(false);
    expect(hasOmlxAuthorizationHeader({})).toBe(false);
    expect(hasOmlxAuthorizationHeader({ "X-Custom": "value" })).toBe(false);
  });
});

describe("resolveOmlxProviderAuthMode", () => {
  it("returns undefined for the synthetic placeholder", () => {
    expect(resolveOmlxProviderAuthMode("omlx-local")).toBeUndefined();
  });

  it("returns undefined for an empty or undefined key", () => {
    expect(resolveOmlxProviderAuthMode(undefined)).toBeUndefined();
    expect(resolveOmlxProviderAuthMode("")).toBeUndefined();
  });

  it("returns api-key for a real configured key", () => {
    expect(resolveOmlxProviderAuthMode("sk-real-key")).toBe("api-key");
  });
});

describe("shouldUseOmlxApiKeyPlaceholder", () => {
  it("is true only when models exist and no key/header is configured", () => {
    expect(shouldUseOmlxApiKeyPlaceholder({ hasModels: true, resolvedApiKey: undefined })).toBe(
      true,
    );
    expect(shouldUseOmlxApiKeyPlaceholder({ hasModels: false, resolvedApiKey: undefined })).toBe(
      false,
    );
    expect(shouldUseOmlxApiKeyPlaceholder({ hasModels: true, resolvedApiKey: "sk-real" })).toBe(
      false,
    );
    expect(
      shouldUseOmlxApiKeyPlaceholder({
        hasModels: true,
        resolvedApiKey: undefined,
        hasAuthorizationHeader: true,
      }),
    ).toBe(false);
  });
});

describe("shouldUseOmlxSyntheticAuth", () => {
  it("synthesizes for a configured provider with models and no explicit auth", () => {
    expect(
      shouldUseOmlxSyntheticAuth({
        api: "openai-completions",
        baseUrl: "http://localhost:8000/v1",
        models: [
          {
            id: "model-a",
            name: "model-a",
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 32768,
            maxTokens: 8192,
          },
        ],
      }),
    ).toBe(true);
  });

  it("does not synthesize without configured models", () => {
    expect(
      shouldUseOmlxSyntheticAuth({
        api: "openai-completions",
        baseUrl: "http://localhost:8000/v1",
        models: [],
      }),
    ).toBe(false);
    expect(shouldUseOmlxSyntheticAuth(undefined)).toBe(false);
  });

  it("does not synthesize when a real API key or Authorization header is already configured", () => {
    const base = {
      api: "openai-completions" as const,
      baseUrl: "http://localhost:8000/v1",
      models: [
        {
          id: "model-a",
          name: "model-a",
          reasoning: false,
          input: ["text" as const],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 32768,
          maxTokens: 8192,
        },
      ],
    };
    expect(shouldUseOmlxSyntheticAuth({ ...base, apiKey: "sk-real-key" })).toBe(false);
    expect(
      shouldUseOmlxSyntheticAuth({ ...base, headers: { Authorization: "Bearer proxy-token" } }),
    ).toBe(false);
  });
});
