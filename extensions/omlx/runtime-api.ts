// Omlx API module exposes the plugin public contract.
export {
  OMLX_DEFAULT_API_KEY_ENV_VAR,
  OMLX_DEFAULT_BASE_URL,
  OMLX_DEFAULT_EMBEDDING_MODEL,
  OMLX_DEFAULT_INFERENCE_BASE_URL,
  OMLX_DEFAULT_MODEL_ID,
  OMLX_LOCAL_API_KEY_PLACEHOLDER,
  OMLX_MODEL_PLACEHOLDER,
  OMLX_PROVIDER_ID,
  OMLX_PROVIDER_LABEL,
} from "./src/defaults.js";
export { discoverOmlxModels, ensureOmlxModelLoaded, fetchOmlxModels } from "./src/models.fetch.js";
export {
  mapOmlxWireEntry,
  mapOmlxWireModelsToConfig,
  normalizeOmlxProviderConfig,
  resolveOmlxContextWindow,
  resolveOmlxInferenceBase,
  resolveOmlxServerBase,
  type OmlxModelBase,
  type OmlxModelWire,
} from "./src/models.js";
export {
  buildOmlxAuthHeaders,
  resolveOmlxConfiguredApiKey,
  resolveOmlxProviderHeaders,
  resolveOmlxRequestContext,
  resolveOmlxRuntimeApiKey,
} from "./src/runtime.js";
