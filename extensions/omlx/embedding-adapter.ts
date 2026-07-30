// Omlx plugin module adapts its remote embeddings runtime to the generic provider contract.
import type {
  EmbeddingInput,
  EmbeddingProvider,
  EmbeddingProviderAdapter,
  EmbeddingProviderCreateOptions,
} from "openclaw/plugin-sdk/embedding-providers";
import type { MemoryEmbeddingProvider } from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";
import { OMLX_PROVIDER_ID } from "./src/defaults.js";
import {
  createOmlxEmbeddingProvider,
  DEFAULT_OMLX_EMBEDDING_MODEL,
} from "./src/embedding-provider.js";

function textFromEmbeddingInput(input: EmbeddingInput): string {
  return typeof input === "string" ? input : input.text;
}

function adaptMemoryEmbeddingProvider(provider: MemoryEmbeddingProvider): EmbeddingProvider {
  return {
    id: provider.id,
    model: provider.model,
    ...(typeof provider.maxInputTokens === "number"
      ? { maxInputTokens: provider.maxInputTokens }
      : {}),
    embed: async (input, options) =>
      await provider.embedQuery(textFromEmbeddingInput(input), { signal: options?.signal }),
    embedBatch: async (inputs, options) =>
      await provider.embedBatch(inputs.map(textFromEmbeddingInput), { signal: options?.signal }),
  };
}

function buildMemoryCreateOptions(options: EmbeddingProviderCreateOptions) {
  return {
    config: options.config,
    agentDir: options.agentDir,
    provider: OMLX_PROVIDER_ID,
    fallback: "none" as const,
    remote: options.remote,
    model: options.model,
    inputType: options.inputType,
    queryInputType: options.queryInputType,
    documentInputType: options.documentInputType,
    outputDimensionality: options.dimensions,
  };
}

export const omlxEmbeddingProviderAdapter: EmbeddingProviderAdapter = {
  id: OMLX_PROVIDER_ID,
  defaultModel: DEFAULT_OMLX_EMBEDDING_MODEL,
  transport: "remote",
  authProviderId: OMLX_PROVIDER_ID,
  create: async (options) => {
    const { provider, client } = await createOmlxEmbeddingProvider(
      buildMemoryCreateOptions(options),
    );
    return {
      provider: adaptMemoryEmbeddingProvider(provider),
      runtime: {
        id: OMLX_PROVIDER_ID,
        cacheKeyData: {
          provider: OMLX_PROVIDER_ID,
          baseUrl: client.baseUrl,
          model: client.model,
        },
      },
    };
  },
};
