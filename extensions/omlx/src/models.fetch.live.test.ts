// Omlx live tests exercise discovery against a real oMLX server.
import { describe, expect, it } from "vitest";
import { OMLX_DEFAULT_BASE_URL, OMLX_DEFAULT_EMBEDDING_MODEL } from "./defaults.js";
import { discoverOmlxModels, fetchOmlxModels, resolveOmlxEmbeddingModel } from "./models.fetch.js";
import { collectOmlxEmbeddingModelIds, mapOmlxWireEntry } from "./models.js";

const LIVE = process.env.OPENCLAW_LIVE_TEST === "1" && process.env.OPENCLAW_LIVE_OMLX === "1";
const BASE_URL = process.env.OPENCLAW_LIVE_OMLX_BASE_URL ?? OMLX_DEFAULT_BASE_URL;
const API_KEY = process.env.OMLX_API_KEY ?? "";

// Discovery is read-only: no model is loaded or unloaded, so this suite never
// mutates the operator's running oMLX instance.
describe.runIf(LIVE)("oMLX live discovery", () => {
  it("reads the model catalog out of the status envelope", async () => {
    const result = await fetchOmlxModels({ baseUrl: BASE_URL, apiKey: API_KEY });

    expect(result.error).toBeUndefined();
    expect(result.reachable).toBe(true);
    expect(result.status).toBe(200);
    // A bare-array reader silently yields zero models against a real server.
    expect(result.models.length).toBeGreaterThan(0);
    for (const entry of result.models) {
      expect(typeof entry.id).toBe("string");
      expect(typeof entry.model_type).toBe("string");
    }
  });

  it("maps every chat model to a budget the runtime can spend", async () => {
    const models = await discoverOmlxModels({ baseUrl: BASE_URL, apiKey: API_KEY, quiet: true });

    expect(models.length).toBeGreaterThan(0);
    for (const model of models) {
      expect(model.maxTokens).toBeGreaterThan(0);
      // oMLX advertises max_tokens as a server-wide ceiling, so an unbounded
      // mapping produces maxTokens above the model's own window.
      expect(model.maxTokens).toBeLessThanOrEqual(model.contextWindow ?? 0);
      expect(model.contextTokens ?? 0).toBeLessThanOrEqual(model.contextWindow ?? 0);
    }
  });

  it("excludes embedding, audio, and helper entries from the chat catalog", async () => {
    const { models: wire } = await fetchOmlxModels({ baseUrl: BASE_URL, apiKey: API_KEY });
    const chatIds = new Set(
      (await discoverOmlxModels({ baseUrl: BASE_URL, apiKey: API_KEY, quiet: true })).map(
        (model) => model.id,
      ),
    );

    const nonChat = wire.filter(
      (entry) =>
        entry.is_helper === true || (entry.model_type !== "llm" && entry.model_type !== "vlm"),
    );
    expect(nonChat.length).toBeGreaterThan(0);
    for (const entry of nonChat) {
      expect(chatIds.has(String(entry.id))).toBe(false);
      expect(mapOmlxWireEntry(entry)).toBeNull();
    }
  });
});

// Embedding models are excluded from the chat catalog, so memory search needs its
// own detection path. These stay read-only: no model is loaded or unloaded.
describe.runIf(LIVE)("oMLX live embedding detection", () => {
  it("advertises at least one embedding model, disjoint from the chat catalog", async () => {
    const { models: wire } = await fetchOmlxModels({ baseUrl: BASE_URL, apiKey: API_KEY });
    const embeddingIds = collectOmlxEmbeddingModelIds(wire);
    const chatIds = new Set(
      (await discoverOmlxModels({ baseUrl: BASE_URL, apiKey: API_KEY, quiet: true })).map(
        (model) => model.id,
      ),
    );

    expect(embeddingIds.length).toBeGreaterThan(0);
    for (const id of embeddingIds) {
      expect(chatIds.has(id)).toBe(false);
    }
  });

  it("resolves a usable embedding model with nothing configured", async () => {
    const resolved = await resolveOmlxEmbeddingModel({ baseUrl: BASE_URL, apiKey: API_KEY });

    expect(resolved.status).toBe("resolved");
    if (resolved.status === "resolved") {
      expect(resolved.modelId.length).toBeGreaterThan(0);
      // Route-safe ids never contain a slash; an `org/repo` value here means the
      // mapper leaked a source_repo_id into the id.
      expect(resolved.modelId).not.toContain("/");
    }
  });

  it("prefers the bundled default when this server advertises it", async () => {
    const { models: wire } = await fetchOmlxModels({ baseUrl: BASE_URL, apiKey: API_KEY });
    if (!collectOmlxEmbeddingModelIds(wire).includes(OMLX_DEFAULT_EMBEDDING_MODEL)) {
      return;
    }
    const resolved = await resolveOmlxEmbeddingModel({
      baseUrl: BASE_URL,
      apiKey: API_KEY,
      preferred: OMLX_DEFAULT_EMBEDDING_MODEL,
    });

    expect(resolved).toMatchObject({ status: "resolved", modelId: OMLX_DEFAULT_EMBEDDING_MODEL });
  });

  it("names the real alternatives for a model the server does not serve", async () => {
    const resolved = await resolveOmlxEmbeddingModel({
      baseUrl: BASE_URL,
      apiKey: API_KEY,
      requested: "mlx-community/definitely-not-installed",
    });

    expect(resolved.status).toBe("absent");
    if (resolved.status === "absent") {
      expect(resolved.available.length).toBeGreaterThan(0);
    }
  });

  it("separates an unreachable server from a missing model", async () => {
    const resolved = await resolveOmlxEmbeddingModel({
      baseUrl: "http://127.0.0.1:59999",
      requested: "anything",
      timeoutMs: 1500,
    });

    expect(resolved.status).toBe("unavailable");
  });
});
