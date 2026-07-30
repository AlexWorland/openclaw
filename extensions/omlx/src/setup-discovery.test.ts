// Omlx tests cover setup discovery behavior.
import type { ModelDefinitionConfig } from "openclaw/plugin-sdk/provider-model-shared";
import { describe, expect, it } from "vitest";
import { OMLX_DEFAULT_MODEL_ID } from "./defaults.js";
import {
  collectAppGuidedOmlxModelIds,
  isOmlxDiscoveryConfigResolutionError,
  resolveOmlxDiscoveryFailure,
  selectDefaultOmlxModelId,
  type OmlxDiscoveryResult,
} from "./setup-discovery.js";

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

function createDiscovery(models: Record<string, unknown>[]): OmlxDiscoveryResult {
  return { reachable: true, status: 200, models } as OmlxDiscoveryResult;
}

describe("resolveOmlxDiscoveryFailure", () => {
  it("reports an unreachable server with a start-the-server hint", () => {
    const failure = resolveOmlxDiscoveryFailure({
      baseUrl: "http://localhost:8000/v1",
      discovery: { reachable: false, models: [] } as OmlxDiscoveryResult,
    });

    expect(failure?.reason).toBe("oMLX not reachable");
    expect(failure?.noteLines[0]).toContain("http://localhost:8000/v1");
  });

  it("reports an HTTP error status distinctly from unreachability", () => {
    const failure = resolveOmlxDiscoveryFailure({
      baseUrl: "http://localhost:8000/v1",
      discovery: { reachable: true, status: 401, models: [] } as OmlxDiscoveryResult,
    });

    expect(failure?.reason).toBe("oMLX discovery failed (401)");
  });

  it("reports an empty usable-model set when only non-chat models exist", () => {
    const failure = resolveOmlxDiscoveryFailure({
      baseUrl: "http://localhost:8000/v1",
      discovery: createDiscovery([{ id: "embed-model", model_type: "embedding" }]),
    });

    expect(failure?.reason).toBe("No oMLX models found");
  });

  it("treats helper-only discovery as having no usable model", () => {
    const failure = resolveOmlxDiscoveryFailure({
      baseUrl: "http://localhost:8000/v1",
      discovery: createDiscovery([{ id: "drafter", model_type: "llm", is_helper: true }]),
    });

    expect(failure?.reason).toBe("No oMLX models found");
  });

  it("returns null when at least one llm or vlm model is usable", () => {
    expect(
      resolveOmlxDiscoveryFailure({
        baseUrl: "http://localhost:8000/v1",
        discovery: createDiscovery([{ id: "model-a", model_type: "llm" }]),
      }),
    ).toBeNull();
    expect(
      resolveOmlxDiscoveryFailure({
        baseUrl: "http://localhost:8000/v1",
        discovery: createDiscovery([{ id: "model-v", model_type: "vlm" }]),
      }),
    ).toBeNull();
  });
});

describe("selectDefaultOmlxModelId", () => {
  it("prefers the bundled default model id when the server exposes it", () => {
    expect(
      selectDefaultOmlxModelId([createModel("other-model"), createModel(OMLX_DEFAULT_MODEL_ID)]),
    ).toBe(OMLX_DEFAULT_MODEL_ID);
  });

  it("falls back to a discovered id when the bundled default is absent", () => {
    expect(selectDefaultOmlxModelId([createModel("only-model")])).toBe("only-model");
  });

  it("returns undefined for an empty catalog", () => {
    expect(selectDefaultOmlxModelId([])).toBeUndefined();
  });
});

describe("collectAppGuidedOmlxModelIds", () => {
  it("includes text LLMs at or above the minimum context window", () => {
    expect(
      collectAppGuidedOmlxModelIds(
        createDiscovery([{ id: "big-llm", model_type: "llm", max_context_window: 32768 }]),
      ),
    ).toEqual(new Set(["big-llm"]));
  });

  it("excludes models below the minimum context window", () => {
    expect(
      collectAppGuidedOmlxModelIds(
        createDiscovery([{ id: "tiny-llm", model_type: "llm", max_context_window: 2048 }]),
      ).size,
    ).toBe(0);
  });

  it("excludes helper models and non-llm model types", () => {
    expect(
      collectAppGuidedOmlxModelIds(
        createDiscovery([
          { id: "drafter", model_type: "llm", max_context_window: 32768, is_helper: true },
          { id: "vision", model_type: "vlm", max_context_window: 32768 },
          { id: "embed", model_type: "embedding", max_context_window: 32768 },
        ]),
      ).size,
    ).toBe(0);
  });
});

describe("isOmlxDiscoveryConfigResolutionError", () => {
  it("matches unresolved omlx apiKey and header secret refs", () => {
    expect(
      isOmlxDiscoveryConfigResolutionError(
        new Error("models.providers.omlx.apiKey: unresolved secret ref"),
      ),
    ).toBe(true);
    expect(
      isOmlxDiscoveryConfigResolutionError(
        new Error("models.providers.omlx.headers.Authorization: unresolved secret ref"),
      ),
    ).toBe(true);
  });

  it("does not match unrelated errors", () => {
    expect(isOmlxDiscoveryConfigResolutionError(new Error("ECONNREFUSED"))).toBe(false);
  });
});
