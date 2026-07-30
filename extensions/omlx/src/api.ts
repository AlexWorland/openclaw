// Omlx API module exposes the plugin public contract.
export {
  OMLX_DEFAULT_API_KEY_ENV_VAR,
  OMLX_DEFAULT_BASE_URL,
  OMLX_DEFAULT_EMBEDDING_MODEL,
  OMLX_DEFAULT_INFERENCE_BASE_URL,
  OMLX_DEFAULT_MODEL_ID,
  OMLX_DOCKER_HOST_BASE_URL,
  OMLX_DOCKER_HOST_INFERENCE_BASE_URL,
  OMLX_LOCAL_API_KEY_PLACEHOLDER,
  OMLX_MODEL_PLACEHOLDER,
  OMLX_PROVIDER_ID,
  OMLX_PROVIDER_LABEL,
} from "./defaults.js";
export {
  buildOmlxModelName,
  type OmlxModelBase,
  type OmlxModelWire,
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
export {
  buildOmlxAuthHeaders,
  resolveOmlxConfiguredApiKey,
  resolveOmlxProviderHeaders,
  resolveOmlxRequestContext,
  resolveOmlxRuntimeApiKey,
} from "./runtime.js";
export {
  configureOmlxNonInteractive,
  discoverOmlxProvider,
  prepareAppGuidedOmlxSetup,
  prepareOmlxDynamicModels,
  promptAndConfigureOmlxInteractive,
} from "./setup.js";
