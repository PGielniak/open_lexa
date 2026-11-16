# Tracing Open Lexa

The shared monitoring stack is managed in `~/src/homelab/kubernetes/observability`.
The receiver and Whisper API enable asynchronous OTLP/HTTP tracing when
`OTEL_EXPORTER_OTLP_ENDPOINT` is set; without it tracing is disabled. The k3s
Deployments point to `http://alloy.monitoring.svc.cluster.local:4318`. The Alloy
Service load-balances across nodes; it does not guarantee a node-local agent.
Local runs remain unchanged unless the environment variable is set.

The receiver emits one `utterance` per accepted complete recording, with
`receive_audio`, `receiver_queue_wait`, `stt` and `llm` spans. Receive
and queue timestamps are derived from the receiver's monotonic clock and mapped
to wall time at worker start; they do not include capture or transport before
the first packet arrives. In dual mode `stt.local` and `stt.cloud` run in
parallel and retain their parent context. For the local HTTP backend, W3C trace
context crosses to `whisper.http`, `whisper.inference`,
`whisper.select_model` and `whisper.transcribe`; the Hailo calls still run on
their original single owner thread. The 32-bit session value is an attribute,
not a globally unique trace ID. Incomplete, rejected and dropped recordings do
not currently produce traces. Firmware capture, retransmissions, application
metrics, and explicit archive spans are not yet instrumented.

The `observability/` Kustomize component provisions the **Open Lexa — przegląd**
Grafana dashboard through the cluster's dashboard sidecar:

```sh
kubectl --context default apply -k observability
```

It shows Kubernetes availability, CPU, memory, throttling and Loki logs. Its
Tempo panel displays traces from the instrumented receiver image;
there are no stage-latency histograms or p50/p95 panels yet. This component does
not modify application deployments or require an image rebuild.

Log records generated inside an active span include `trace_id` and `span_id`.
Receiver logs also include the chosen `transcript`, MCP `mcp_call` with tool
name and arguments, and `mcp_result` with status and execution time. Each line
includes the recording `session` for correlation. Timing fields in milliseconds:
`stt_backend` for each local/cloud STT call, `stt_total` for the parallel STT
stage, `mcp_discovery`, `llm_request` for the model HTTP call alone,
`mcp_result` for tool execution, `agent_total` for discovery + LLM + tool, and
`processing_total` for the receiver worker (excluding reception and queue wait;
archive writing is asynchronous). Failures and a decision not to call a tool
are logged as well. Exceptions caught by STT, LLM, Sonos/Spotify and MCP are
logged with Python tracebacks and chained causes. A tool that returns `failed`
or `skipped` without raising logs its reason in `mcp_result`; there is no Python
traceback unless an exception was raised locally (remote MCP errors may only
provide a message). In Grafana Explore, use
`{namespace="open-lexa", container="receiver"}` and filter by `session=`.

These logs contain transcripts and tool arguments and may contain personal data.
Keys such as `password`, `token`, `api_key` and `secret` are masked recursively
in logged tool arguments, but values under other names and service errors may
still contain sensitive information. Protect access to Grafana/Loki and apply
an appropriate retention and privacy policy. Never put
session or trace IDs in metric labels. Trace payloads contain no audio, token or
transcript, but existing exception messages may contain service error details.
Telemetry export uses a bounded background batch processor and a short request
timeout; observability outages must not stop command handling. Verify on the
target cluster after deploying new images: send a test WAV and inspect a trace in
Grafana Explore. A Tempo trace alone does not establish end-user latency before
the first packet or audible Sonos playback completion.
