// Omlx live tests exercise discovery against a real oMLX server.
import { describe, expect, it } from "vitest";
import { OMLX_DEFAULT_BASE_URL } from "./defaults.js";
import { discoverOmlxModels, fetchOmlxModels } from "./models.fetch.js";
import { mapOmlxWireEntry } from "./models.js";

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
