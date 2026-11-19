# hailo-llm — hailo-ollama + Gemma4-E2B on raspberry02

`open-lexa-llm` (Hailo-10H, port 8000, Ollama API + `/v1/chat/completions`) and
Open WebUI (`open-webui/`, tailnet only at `https://hailo-chat.<tailnet>.ts.net`).

## Build

`hailo-llm/vendor/` is git-ignored (licensed Hailo binary). Refresh it from a node
with `hailo-gen-ai-model-zoo` installed, then follow the Tekton steps in
`build/pipeline.yaml` (context ConfigMap must be applied **server-side**):

```sh
scp raspberry02:/usr/bin/hailo-ollama hailo-llm/vendor/hailo-ollama
scp -r raspberry02:/usr/share/hailo-ollama/models hailo-llm/vendor/share/hailo-ollama/
```

## Deploy

```sh
kubectl label node raspberry02 hailo.ai/device=hailo10h   # runs the device plugin there
kubectl apply -k hailo-llm/deploy
kubectl apply -k open-webui/deploy   # needs Secret open-webui (see its kustomization)
```

The model (~3.3 GB) lives in a hostPath on raspberry02
(`/home/patryk/.local/share/hailo-ollama`), not in the image. Pull it with
`POST /api/pull {"model":"gemma4:e2b"}` (needs `Content-Type: application/json`).

## Known limits (hailo-ollama 5.5.0)

- Gemma4 tool calls come back as raw text in `message.content`
  (`<|tool_call>call:name{...}<tool_call|>`), **not** in `tool_calls`. The
  `tool-adapter` sidecar (`adapter/gemma_tool_adapter.py`, port 8001, what the
  Service points at) parses that into `tool_calls` for `/v1/chat/completions` and
  `/api/chat`; requests without `tools` are passed through. Tests: `hailo-llm/tests`.
- `role: tool` follow-up messages return an empty reply (the server clears context
  when tools are present), so only single-shot tool calls work.
- The 1.5B-class quality shows: e.g. "2,5 minuty" came back as `minutes:120`.
  Validate arguments (the receiver already does).
- 2048-token context, ~5 tok/s, ~17 s to load the model on first request.
- One request at a time; do not run another Hailo workload on this node.
