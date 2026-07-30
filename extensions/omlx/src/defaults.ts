/** Shared oMLX defaults used by setup, runtime discovery, stream, and embeddings paths. */
export const OMLX_PROVIDER_ID = "omlx";
export const OMLX_PROVIDER_LABEL = "oMLX";
export const OMLX_DEFAULT_BASE_URL = "http://localhost:8000";
export const OMLX_DEFAULT_INFERENCE_BASE_URL = `${OMLX_DEFAULT_BASE_URL}/v1`;
export const OMLX_DOCKER_HOST_BASE_URL = "http://host.docker.internal:8000";
export const OMLX_DOCKER_HOST_INFERENCE_BASE_URL = `${OMLX_DOCKER_HOST_BASE_URL}/v1`;
export const OMLX_DEFAULT_API_KEY_ENV_VAR = "OMLX_API_KEY";
export const OMLX_LOCAL_API_KEY_PLACEHOLDER = "omlx-local";
// oMLX model ids are route-safe and never contain `/`: a locally discovered model
// uses its directory name, and an HF-cache model encodes `org/repo` as `org--repo`
// (omlx/model_discovery.py `_decode_hf_cache_model_id`). A `mlx-community/...` ref
// is a `source_repo_id`, not an id, and cannot be loaded or used for inference.
export const OMLX_MODEL_PLACEHOLDER = "Qwen3-8B-4bit";
export const OMLX_DEFAULT_MODEL_ID = "Qwen3-8B-4bit";
export const OMLX_DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-0.6B-4bit-DWQ";
// oMLX loads models asynchronously; cold loads of large models can take a while.
export const OMLX_LOAD_TIMEOUT_MS = 120_000;
export const OMLX_DISCOVERY_TIMEOUT_MS = 8_000;
