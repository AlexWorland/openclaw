// Omlx tests cover models behavior.
import { describe, expect, it } from "vitest";
import {
  buildOmlxModelName,
  mapOmlxWireEntry,
  mapOmlxWireModelsToConfig,
  normalizeOmlxConfiguredCatalogEntries,
  normalizeOmlxConfiguredCatalogEntry,
  normalizeOmlxModelId,
  normalizeOmlxProviderConfig,
  resolveOmlxContextWindow,
  resolveOmlxInferenceBase,
  resolveOmlxServerBase,
} from "./models.js";

describe("resolveOmlxServerBase / resolveOmlxInferenceBase", () => {
  it("defaults to the local oMLX server when nothing is configured", () => {
    expect(resolveOmlxServerBase()).toBe("http://localhost:8000");
    expect(resolveOmlxInferenceBase()).toBe("http://localhost:8000/v1");
  });

  it("strips a trailing /v1 from a pasted inference URL", () => {
    expect(resolveOmlxServerBase("http://localhost:8000/v1")).toBe("http://localhost:8000");
    expect(resolveOmlxInferenceBase("http://localhost:8000/v1")).toBe("http://localhost:8000/v1");
  });

  it("accepts a bare host:port without an explicit scheme", () => {
    expect(resolveOmlxServerBase("localhost:9000")).toBe("http://localhost:9000");
  });

  it("preserves an explicit https scheme", () => {
    expect(resolveOmlxServerBase("https://omlx.internal:8443")).toBe("https://omlx.internal:8443");
  });

  it("falls back to the default only when normalization yields an empty string", () => {
    expect(resolveOmlxServerBase("///")).toBe("http://localhost:8000");
  });

  it("passes through an unparsable value best-effort rather than silently defaulting", () => {
    expect(resolveOmlxServerBase("::not a url::")).toBe("::not a url::");
  });
});

describe("normalizeOmlxProviderConfig", () => {
  it("canonicalizes the base URL to the inference form and defaults allowPrivateNetwork", () => {
    const result = normalizeOmlxProviderConfig({
      baseUrl: "http://localhost:8000",
      api: "openai-completions",
      models: [],
    });
    expect(result.baseUrl).toBe("http://localhost:8000/v1");
    expect(result.request).toEqual({ allowPrivateNetwork: true });
  });

  it("leaves an unset baseUrl untouched", () => {
    const provider = { baseUrl: "", api: "openai-completions" as const, models: [] };
    expect(normalizeOmlxProviderConfig(provider)).toBe(provider);
  });

  it("preserves an explicit allowPrivateNetwork override", () => {
    const result = normalizeOmlxProviderConfig({
      baseUrl: "http://localhost:8000",
      api: "openai-completions",
      models: [],
      request: { allowPrivateNetwork: false },
    });
    expect(result.request).toEqual({ allowPrivateNetwork: false });
  });
});

describe("normalizeOmlxConfiguredCatalogEntry(ies)", () => {
  it("normalizes a well-formed entry", () => {
    expect(
      normalizeOmlxConfiguredCatalogEntry({
        id: "Qwen3-8B-4bit",
        contextWindow: 32768,
        reasoning: true,
        input: ["text", "image", "not-a-real-modality"],
      }),
    ).toEqual({
      id: "Qwen3-8B-4bit",
      name: "Qwen3-8B-4bit",
      contextWindow: 32768,
      contextTokens: undefined,
      reasoning: true,
      input: ["text", "image"],
      compat: undefined,
    });
  });

  it("rejects entries without a usable id", () => {
    expect(normalizeOmlxConfiguredCatalogEntry({})).toBeNull();
    expect(normalizeOmlxConfiguredCatalogEntry({ id: "  " })).toBeNull();
    expect(normalizeOmlxConfiguredCatalogEntry(null)).toBeNull();
  });

  it("filters a non-array models input down to an empty list", () => {
    expect(normalizeOmlxConfiguredCatalogEntries("not-an-array")).toEqual([]);
    expect(normalizeOmlxConfiguredCatalogEntries([{ id: "a" }, {}, { id: "b" }])).toHaveLength(2);
  });
});

describe("normalizeOmlxModelId", () => {
  it("strips leading and trailing slashes and whitespace", () => {
    expect(normalizeOmlxModelId("  /Qwen3-8B-4bit/  ")).toBe("Qwen3-8B-4bit");
  });
});

describe("resolveOmlxContextWindow", () => {
  it("prefers max_context_window over model_context_length", () => {
    expect(
      resolveOmlxContextWindow({ max_context_window: 8192, model_context_length: 262144 }),
    ).toBe(8192);
  });

  it("falls back to model_context_length when max_context_window is missing", () => {
    expect(resolveOmlxContextWindow({ model_context_length: 4096 })).toBe(4096);
  });

  it("falls back to the self-hosted default when both fields are missing", () => {
    expect(resolveOmlxContextWindow({})).toBeGreaterThan(0);
  });
});

describe("buildOmlxModelName", () => {
  it("returns the plain display name with no capability tags", () => {
    expect(
      buildOmlxModelName({
        displayName: "Qwen3-8B",
        vision: false,
        reasoning: false,
        loaded: false,
      }),
    ).toBe("Qwen3-8B");
  });

  it("appends vision/reasoning/loaded tags", () => {
    expect(
      buildOmlxModelName({ displayName: "Qwen3-8B", vision: true, reasoning: true, loaded: true }),
    ).toBe("Qwen3-8B (vision, reasoning, loaded)");
  });
});

describe("mapOmlxWireEntry", () => {
  const baseEntry = {
    id: "Qwen3-8B-4bit",
    model_type: "llm",
    max_context_window: 32768,
    model_context_length: 32768,
    max_tokens: 8192,
    thinking_default: false,
    is_helper: false,
  };

  it("maps a standard LLM entry", () => {
    expect(mapOmlxWireEntry(baseEntry)).toMatchObject({
      id: "Qwen3-8B-4bit",
      displayName: "Qwen3-8B-4bit",
      vision: false,
      reasoning: false,
      loaded: false,
      input: ["text"],
      contextWindow: 32768,
      maxTokens: 8192,
    });
  });

  it("derives a friendly display name from source_repo_id", () => {
    expect(
      mapOmlxWireEntry({ ...baseEntry, source_repo_id: "mlx-community/Qwen3-8B-4bit" })
        ?.displayName,
    ).toBe("Qwen3-8B-4bit");
  });

  it("marks reasoning true when thinking_default is true", () => {
    expect(mapOmlxWireEntry({ ...baseEntry, thinking_default: true })?.reasoning).toBe(true);
  });

  it("marks vision true and sets image input for vlm entries", () => {
    const result = mapOmlxWireEntry({ ...baseEntry, model_type: "vlm" });
    expect(result?.vision).toBe(true);
    expect(result?.input).toEqual(["text", "image"]);
  });

  it("marks loaded true when the discovery entry reports loaded", () => {
    expect(mapOmlxWireEntry({ ...baseEntry, loaded: true })?.loaded).toBe(true);
  });

  it("excludes embedding, audio_tts, and audio_stt entries from the chat catalog", () => {
    for (const model_type of ["embedding", "audio_tts", "audio_stt"]) {
      expect(mapOmlxWireEntry({ ...baseEntry, model_type })).toBeNull();
    }
  });

  it("excludes helper (speculative-decoding draft) models", () => {
    expect(mapOmlxWireEntry({ ...baseEntry, is_helper: true })).toBeNull();
  });

  it("excludes entries missing an id or an unrecognized model_type", () => {
    expect(mapOmlxWireEntry({ ...baseEntry, id: undefined })).toBeNull();
    expect(mapOmlxWireEntry({ ...baseEntry, id: "" })).toBeNull();
    expect(mapOmlxWireEntry({ ...baseEntry, model_type: undefined })).toBeNull();
  });

  it("does not throw on a malformed entry", () => {
    expect(() => mapOmlxWireEntry({})).not.toThrow();
    expect(mapOmlxWireEntry({})).toBeNull();
  });
});

describe("mapOmlxWireModelsToConfig", () => {
  it("maps and filters a batch of wire entries", () => {
    const entries = mapOmlxWireModelsToConfig([
      { id: "model-a", model_type: "llm" },
      { id: "drafter", model_type: "llm", is_helper: true },
      { id: "embed-model", model_type: "embedding" },
    ]);
    expect(entries).toHaveLength(1);
    expect(entries[0]?.id).toBe("model-a");
  });
});
