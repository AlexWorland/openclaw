// Omlx plugin entrypoint registers its OpenClaw integration.
import {
  definePluginEntry,
  type OpenClawPluginApi,
  type ProviderAuthContext,
  type ProviderAuthMethod,
  type ProviderAuthMethodNonInteractiveContext,
  type ProviderAuthResult,
  type ProviderRuntimeModel,
} from "openclaw/plugin-sdk/plugin-entry";
import type { OpenClawConfig } from "openclaw/plugin-sdk/plugin-entry";
import {
  CUSTOM_LOCAL_AUTH_MARKER,
  normalizeOptionalSecretInput,
} from "openclaw/plugin-sdk/provider-auth";
import { omlxEmbeddingProviderAdapter } from "./embedding-adapter.js";
import {
  OMLX_DEFAULT_API_KEY_ENV_VAR,
  OMLX_DEFAULT_INFERENCE_BASE_URL,
  OMLX_LOCAL_API_KEY_PLACEHOLDER,
  OMLX_PROVIDER_LABEL,
} from "./src/defaults.js";
import {
  normalizeOmlxConfiguredCatalogEntries,
  normalizeOmlxProviderConfig,
  resolveOmlxInferenceBase,
} from "./src/models.js";
import { shouldUseOmlxSyntheticAuth } from "./src/provider-auth.js";
import { wrapOmlxInferencePreload } from "./src/stream.js";

const PROVIDER_ID = "omlx";
// Intentional: dynamic models are cached per oMLX endpoint (`baseUrl`) only.
const cachedDynamicModels = new Map<string, ProviderRuntimeModel[]>();

type OmlxNonInteractiveValidationContext = Parameters<
  NonNullable<ProviderAuthMethod["validateNonInteractive"]>
>[0];

async function validateOmlxNonInteractive(
  ctx: OmlxNonInteractiveValidationContext,
): Promise<boolean> {
  const configuredBaseUrl = normalizeOptionalSecretInput(ctx.opts.customBaseUrl);
  const baseUrl = resolveOmlxInferenceBase(configuredBaseUrl || OMLX_DEFAULT_INFERENCE_BASE_URL);
  const providerApiKey = normalizeOptionalSecretInput(ctx.opts.omlxApiKey);
  const resolvedApiKey = await ctx.resolveApiKey({
    provider: PROVIDER_ID,
    flagValue: providerApiKey ?? normalizeOptionalSecretInput(ctx.opts.customApiKey),
    flagName: providerApiKey === undefined ? "--custom-api-key" : "--omlx-api-key",
    envVar: OMLX_DEFAULT_API_KEY_ENV_VAR,
    envVarName: OMLX_DEFAULT_API_KEY_ENV_VAR,
    required: false,
  });

  // A reset preflight may inspect the model catalog but must never invoke
  // setup, write credentials, load a model, or mutate the model server.
  const { fetchOmlxModels } = await import("./src/models.fetch.js");
  const discovery = await fetchOmlxModels({
    baseUrl,
    apiKey: resolvedApiKey?.key ?? OMLX_LOCAL_API_KEY_PLACEHOLDER,
    timeoutMs: 5000,
  });
  if (!discovery.reachable) {
    ctx.runtime.error(
      `oMLX could not be reached at ${baseUrl}.\nStart the oMLX server and re-run setup.`,
    );
    ctx.runtime.exit(1);
    return false;
  }
  if (discovery.status !== undefined && discovery.status >= 400) {
    ctx.runtime.error(
      `oMLX returned HTTP ${discovery.status} while listing models at ${baseUrl}.\nCheck the base URL and API key, then re-run setup.`,
    );
    ctx.runtime.exit(1);
    return false;
  }

  const availableModels = discovery.models
    .filter((model) => model.model_type === "llm" || model.model_type === "vlm")
    .filter((model) => model.is_helper !== true)
    .map((model) => (typeof model.id === "string" ? model.id.trim() : ""))
    .filter((model): model is string => Boolean(model));
  const requestedModel = normalizeOptionalSecretInput(ctx.opts.customModelId);
  if (requestedModel && !availableModels.includes(requestedModel)) {
    ctx.runtime.error(
      `oMLX model ${requestedModel} was not found at ${baseUrl}.\nAvailable models: ${availableModels.join(", ")}`,
    );
    ctx.runtime.exit(1);
    return false;
  }
  if (availableModels.length === 0) {
    ctx.runtime.error(
      `No oMLX LLM/VLM models were found at ${baseUrl}.\nLoad at least one model in oMLX, then re-run setup.`,
    );
    ctx.runtime.exit(1);
    return false;
  }

  return true;
}

function resolveOmlxAugmentedCatalogEntries(config: OpenClawConfig | undefined) {
  if (!config) {
    return [];
  }
  return normalizeOmlxConfiguredCatalogEntries(config.models?.providers?.[PROVIDER_ID]?.models).map(
    (entry) => ({
      provider: PROVIDER_ID,
      id: entry.id,
      name: entry.name ?? entry.id,
      compat: { ...entry.compat, supportsUsageInStreaming: true },
      contextWindow: entry.contextWindow,
      contextTokens: entry.contextTokens,
      reasoning: entry.reasoning,
      input: entry.input,
    }),
  );
}

/** Lazily loads setup helpers so provider wiring stays lightweight at startup. */
async function loadProviderSetup() {
  return await import("./src/setup.js");
}

export default definePluginEntry({
  id: PROVIDER_ID,
  name: "oMLX Provider",
  description: "Bundled oMLX provider plugin",
  register(api: OpenClawPluginApi) {
    api.registerEmbeddingProvider(omlxEmbeddingProviderAdapter);
    api.registerProvider({
      id: PROVIDER_ID,
      label: OMLX_PROVIDER_LABEL,
      docsPath: "/providers/omlx",
      envVars: [OMLX_DEFAULT_API_KEY_ENV_VAR],
      auth: [
        {
          id: "custom",
          label: OMLX_PROVIDER_LABEL,
          hint: "Local Apple Silicon (MLX) inference server",
          kind: "custom",
          appGuidedSetup: {
            detect: async (ctx) => {
              const providerSetup = await loadProviderSetup();
              const result = await providerSetup.prepareAppGuidedOmlxSetup(ctx);
              if (!result?.defaultModel) {
                return null;
              }
              const provider = result.configPatch?.models?.providers?.[PROVIDER_ID];
              return {
                modelRef: result.defaultModel,
                detail: `${result.defaultModel.slice(`${PROVIDER_ID}/`.length)} at ${provider?.baseUrl ?? OMLX_PROVIDER_LABEL}`,
              };
            },
            prepare: async (ctx) => {
              const providerSetup = await loadProviderSetup();
              return await providerSetup.prepareAppGuidedOmlxSetup(ctx);
            },
          },
          run: async (ctx: ProviderAuthContext): Promise<ProviderAuthResult> => {
            const providerSetup = await loadProviderSetup();
            return await providerSetup.promptAndConfigureOmlxInteractive({
              config: ctx.config,
              agentDir: ctx.agentDir,
              prompter: ctx.prompter,
              secretInputMode: ctx.secretInputMode,
              allowSecretRefPrompt: ctx.allowSecretRefPrompt,
            });
          },
          validateNonInteractive: validateOmlxNonInteractive,
          runNonInteractive: async (ctx: ProviderAuthMethodNonInteractiveContext) => {
            const providerSetup = await loadProviderSetup();
            return await providerSetup.configureOmlxNonInteractive(ctx);
          },
        },
      ],
      catalog: {
        // Run after early providers so local oMLX detection does not dominate resolution.
        order: "late",
        run: async (ctx) => {
          const providerSetup = await loadProviderSetup();
          return await providerSetup.discoverOmlxProvider(ctx);
        },
      },
      resolveSyntheticAuth: ({ providerConfig }) => {
        if (!shouldUseOmlxSyntheticAuth(providerConfig)) {
          return undefined;
        }
        return {
          apiKey: CUSTOM_LOCAL_AUTH_MARKER,
          source: "models.providers.omlx (synthetic local key)",
          mode: "api-key" as const,
        };
      },
      shouldDeferSyntheticProfileAuth: ({ resolvedApiKey }) =>
        resolvedApiKey?.trim() === OMLX_LOCAL_API_KEY_PLACEHOLDER ||
        resolvedApiKey?.trim() === CUSTOM_LOCAL_AUTH_MARKER,
      normalizeConfig: ({ providerConfig }) => normalizeOmlxProviderConfig(providerConfig),
      prepareDynamicModel: async (ctx) => {
        const providerSetup = await loadProviderSetup();
        cachedDynamicModels.set(
          ctx.providerConfig?.baseUrl ?? "",
          await providerSetup.prepareOmlxDynamicModels(ctx),
        );
      },
      resolveDynamicModel: (ctx) =>
        cachedDynamicModels
          .get(ctx.providerConfig?.baseUrl ?? "")
          ?.find((model) => model.id === ctx.modelId),
      augmentModelCatalog: (ctx) => resolveOmlxAugmentedCatalogEntries(ctx.config),
      buildUnknownModelHint: () =>
        `${OMLX_PROVIDER_LABEL} auto-detects local models once the server is reachable. ` +
        `If it isn't picked up, set models.providers.omlx.baseUrl (default ${OMLX_DEFAULT_INFERENCE_BASE_URL}) ` +
        'or run "openclaw configure". See: https://docs.openclaw.ai/providers/omlx',
      wrapStreamFn: wrapOmlxInferencePreload,
      wizard: {
        setup: {
          choiceId: PROVIDER_ID,
          choiceLabel: OMLX_PROVIDER_LABEL,
          choiceHint: "Local Apple Silicon (MLX) inference server",
          groupId: PROVIDER_ID,
          groupLabel: OMLX_PROVIDER_LABEL,
          groupHint: "Self-hosted Apple Silicon (MLX) models",
          methodId: "custom",
        },
        modelPicker: {
          label: "oMLX (custom)",
          hint: "Detect models from oMLX /v1/models/status",
          methodId: "custom",
        },
      },
    });
  },
});
