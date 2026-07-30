---
summary: "Run OpenClaw with oMLX (local Apple Silicon / MLX inference server)"
read_when:
  - You want to run OpenClaw against a local oMLX server on Apple Silicon
  - You want oMLX's model discovery, auto-load, and embeddings support
title: "oMLX"
---

oMLX is a local inference server for Apple Silicon (MLX) that exposes an **OpenAI-compatible** HTTP API plus a richer `/v1/models/status` discovery endpoint (model type, context window, reasoning defaults, load state). OpenClaw connects using the `openai-completions` API, auto-discovers models, and loads a model before the first inference or embedding call against it.

| Property         | Value                                       |
| ---------------- | ------------------------------------------- |
| Provider ID      | `omlx`                                      |
| API              | `openai-completions` (OpenAI-compatible)    |
| Auth             | None required for localhost (synthetic key) |
| Optional auth    | `OMLX_API_KEY` env var / `--omlx-api-key`   |
| Default base URL | `http://localhost:8000/v1`                  |
| Model preload    | Automatic before first inference/embedding  |

## Getting started

<Steps>
  <Step title="Start the oMLX server">
    Your base URL must expose `/v1` endpoints (`/v1/models/status`, `/v1/chat/completions`, `/v1/embeddings`). oMLX listens on port 8000 by default:

    ```text
    http://localhost:8000/v1
    ```

  </Step>
  <Step title="Run onboarding">
    ```bash
    openclaw onboard
    ```

    When oMLX is already running with a loaded model, guided setup detects it and proposes a default model without prompting for a base URL or key. Otherwise pick **oMLX** from the provider list and accept the default base URL.

  </Step>
  <Step title="Verify the model is available">
    ```bash
    openclaw models list --provider omlx
    ```
  </Step>
</Steps>

<Tip>
For non-interactive setup (CI, scripting), pass the base URL and model directly. No API key is required for a local, unauthenticated oMLX server:

```bash
openclaw onboard --non-interactive \
  --mode local \
  --auth-choice omlx \
  --custom-base-url "http://localhost:8000/v1" \
  --custom-model-id "Qwen3-8B-4bit"
```

Omit `--custom-model-id` to let setup pick the discovered default model. If your server does enforce auth, add `--omlx-api-key "$OMLX_API_KEY"`.

</Tip>

<Note>
Running OpenClaw in Docker against an oMLX server on the host? Set `OPENCLAW_DOCKER_SETUP=1` before onboarding and setup defaults to `http://host.docker.internal:8000/v1` instead of `localhost`.
</Note>

## Model discovery and auto-load

When `models.providers.omlx` is **not** defined, OpenClaw queries `GET http://localhost:8000/v1/models/status` and maps the response into model entries. Discovery uses oMLX's richer metadata:

- `model_type: "llm"` and `"vlm"` become chat models (`"vlm"` gets `input: ["text", "image"]`).
- `model_type: "embedding"`, `"audio_tts"`, and `"audio_stt"` are excluded from the chat catalog.
- `is_helper: true` entries (speculative-decoding draft models) are excluded entirely.
- `thinking_default: true` sets the model's `reasoning` capability.
- `max_context_window` (falling back to `model_context_length`) sets `contextWindow`.

Before the first inference or embedding call against a model, OpenClaw checks `loaded` in the discovery response and, if needed, calls `POST /v1/models/{id}/load`. If the load fails, OpenClaw logs a warning, backs off retrying that model for a while, and proceeds to inference anyway — the model may already be loaded through oMLX's own tooling.

Model names in the catalog carry state tags from discovery, so `openclaw models list --provider omlx` shows entries like `Qwen3-8B-4bit (vision, reasoning, loaded)`.

### Model ids

oMLX ids never contain `/`, because the id goes into the `/v1/models/{id}/load` route:

- A model in one of your oMLX model directories uses its **directory name** — `Qwen3-8B-4bit`.
- A model from the Hugging Face cache encodes the repo with **double dashes** — HF repo `mlx-community/Qwen3-8B-4bit` becomes id `mlx-community--Qwen3-8B-4bit`.

Discovery reports the original repo as `source_repo_id`, which OpenClaw uses only to build a friendlier display name. Configure and select models by id, never by repo.

<Note>
If you set `models.providers.omlx` explicitly, OpenClaw uses your declared models. A model id that is not in your list still resolves when oMLX advertises it, because the provider also consults live discovery at request time.
</Note>

To disable the preload step entirely and let oMLX handle unloaded models itself:

```json5
{
  models: {
    providers: {
      omlx: {
        params: { preload: false },
      },
    },
  },
}
```

## Explicit configuration

Configure explicitly when oMLX runs on a different host or port, you want to pin `contextWindow`/`maxTokens`, or you connect to a trusted loopback, LAN, or Tailscale endpoint:

```json5
{
  models: {
    providers: {
      omlx: {
        baseUrl: "http://localhost:8000/v1",
        api: "openai-completions",
        timeoutSeconds: 300, // Optional: extend request timeout for slow local models
        models: [
          {
            id: "Qwen3-8B-4bit",
            name: "Qwen3 8B (oMLX)",
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 32768,
            maxTokens: 8192,
          },
        ],
      },
    },
  },
}
```

## Embeddings

oMLX is also a memory-search embedding provider. Embedding models are deliberately absent from the chat catalog, so memory search detects them separately: OpenClaw reads `model_type: "embedding"` entries from the same `/v1/models/status` response and loads the selected one before the first embedding request.

Naming a model is optional. Leave `embeddingModel` unset and OpenClaw picks one from live discovery, preferring `Qwen3-Embedding-0.6B-4bit-DWQ` when your server has it:

```json5
{
  memorySearch: {
    embeddingProvider: "omlx",
  },
}
```

Set it explicitly to pin a specific model:

```json5
{
  memorySearch: {
    embeddingProvider: "omlx",
    embeddingModel: "Qwen3-Embedding-0.6B-4bit-DWQ",
  },
}
```

If the model you name is not one your server serves, startup fails with the ids that _are_ available rather than a generic request error. A model oMLX classifies as something other than `embedding` is still accepted when you name it explicitly, so you can point memory search at any advertised model.

<Note>
An unreachable oMLX server is treated differently from a missing model: OpenClaw keeps the configured (or default) embedding model, logs a warning, and lets the embedding call surface the live error. Only a server that answers and does not serve the model is a configuration error.
</Note>

## Prompt caching

oMLX is an OpenAI-compatible proxy, not the native OpenAI endpoint, so `prompt_cache_key` is not sent automatically. Set `compat.supportsPromptCacheKey: true` on the model row to opt in if your oMLX version supports it.

## Troubleshooting

<AccordionGroup>
  <Accordion title="Server not reachable">
    Check that oMLX is running and accessible:

    ```bash
    curl http://localhost:8000/v1/models/status
    ```

    If you see a connection error, verify the host and port. OpenClaw trusts the exact configured `models.providers.omlx.baseUrl` origin for guarded model requests on loopback, LAN, and Tailscale endpoints.

  </Accordion>

  <Accordion title="No models discovered">
    If `models.providers.omlx` is defined, OpenClaw uses only your declared models. Otherwise, confirm the server is reachable and has at least one non-helper `llm`/`vlm` model.
  </Accordion>

  <Accordion title="Model load is slow or times out">
    Large models can take a while to load into unified memory on the first request. OpenClaw's preload backs off exponentially after repeated failures for the same model rather than retrying every request; check the oMLX server logs for the underlying load error.
  </Accordion>

  <Accordion title="Remote oMLX deployment requires a real API key">
    The synthetic local key only applies to localhost. If `baseUrl` points at a non-localhost host, set a real `apiKey` under `models.providers.omlx`.
  </Accordion>
</AccordionGroup>

<Warning>
More help: [Troubleshooting](/help/troubleshooting) and [FAQ](/help/faq).
</Warning>

## Related

<CardGroup cols={2}>
  <Card title="Model selection" href="/concepts/model-providers" icon="layers">
    Choosing providers, model refs, and failover behavior.
  </Card>
  <Card title="LM Studio" href="/providers/lmstudio" icon="cpu">
    Another local Apple Silicon / GGUF provider with a similar auto-load flow.
  </Card>
  <Card title="Local model services" href="/gateway/local-model-services" icon="server">
    Starting local model servers on demand.
  </Card>
  <Card title="Troubleshooting" href="/help/troubleshooting" icon="wrench">
    Common issues and how to resolve them.
  </Card>
</CardGroup>
