# Open Lexa on k3s

The receiver is a UDP service plus an HTTP health endpoint. It sends canonical
WAV recordings to a configurable OpenAI-compatible transcription API. By
default it uses **dual** mode: the local Whisper API, which runs Whisper on
Hailo in a **separate** Deployment and leaves the receiver without a Hailo
device, plus cloud `gpt-transcribe`. Each component keeps its manifests in its
own `deploy/` directory:

```
udp-receiver/deploy/      namespace, receiver ConfigMap, Deployment, Services
udp-receiver/deploy/secret.example.yaml   secret shape (never commit real values)
whisper-api/deploy/       Hailo Whisper API Deployment, ConfigMap, ClusterIP Service
admin-dashboard/deploy/   admin UI Deployment, PVC, Service, Tailscale Ingress
```

Each `kustomization.yaml` pins its image by digest. Building the images:
[`BUILD.md`](BUILD.md).

## Verified cluster behaviour (2026-10-29)

Checked on a four-node k3s `v1.36.5+k3s1` cluster (one amd64 node and three
ARM64 Raspberry Pi nodes, joined over Tailscale):

- k3s ServiceLB (`klipper-lb`) supports **UDP LoadBalancer** services.
- The `svclb` pod publishes the service port as a **hostPort on every node**.
  A UDP packet sent from the LAN to one node's LAN address reached the pod;
  packets sent to the Tailscale IP of all four nodes also reached the pod,
  including cross-node forwarding.
- The `Service.status.loadBalancer.ingress` values are the **Tailscale node
  IPs** because k3s uses `--node-ip=<tailscale IP>`. That does **not** limit
  the hostPort listener: the LAN IPv4 of any Ready node also works. The ESP32
  therefore does not need Tailscale.
- ServiceLB **SNATs** LAN traffic: the pod sees the `svclb` pod/overlay IP,
  not the ESP32 address. Do not rely on `--device-ip` with the LoadBalancer.
  The 16-byte shared token remains the authentication boundary.
- No host firewall blocked UDP on the tested node. If your nodes run `ufw` or
  nftables rules, allow the UDP port from the ESP32/Wi-Fi subnet.

## Create the token/API secret

The token must match `RECEIVER_TOKEN` in
`firmware/src/OpenLexaNano/Config.h` (32 hex characters).

```sh
kubectl apply -f udp-receiver/deploy/namespace.yaml

read -r -p 'OpenAI API key (required for dual/openai; empty only for local-only http): ' OPENAI_KEY
kubectl -n open-lexa create secret generic open-lexa-secrets \
  --from-literal=OPEN_LEXA_TOKEN="$(openssl rand -hex 16)" \
  --from-literal=OPENAI_API_KEY="$OPENAI_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -
unset OPENAI_KEY
```

The generated token is printed by the command above only if you echo it; to
read it back later use:

```sh
kubectl -n open-lexa get secret open-lexa-secrets \
  -o jsonpath='{.data.OPEN_LEXA_TOKEN}' | base64 -d; echo
```

Do not commit `secret.example.yaml` values or a real Secret manifest.

## Deploy

The receiver kustomization owns the `open-lexa` namespace, so apply it first.
The receiver also reads the `open-lexa-smb` Secret (see "SMB archive").

```sh
kubectl apply -k udp-receiver/deploy
kubectl apply -k whisper-api/deploy
kubectl -n open-lexa rollout status deployment/open-lexa-whisper --timeout=180s
kubectl -n open-lexa rollout status deployment/open-lexa --timeout=180s
kubectl -n open-lexa get pods,svc -o wide
```

Check health:

```sh
kubectl -n open-lexa port-forward svc/open-lexa-health 8080:8080
curl -s http://127.0.0.1:8080/healthz
curl -s http://127.0.0.1:8080/readyz
kubectl -n open-lexa logs deployment/open-lexa -c receiver
```

## LAN ingress for the ESP32

Find the LAN address of any Ready node. `kubectl get nodes -o wide` shows the
Tailscale `INTERNAL-IP`; use the host's LAN address instead, for example:

| Node | Example LAN address |
| --- | --- |
| node-a | `192.168.1.50` (wlan0) |
| node-b | `192.168.1.51` (wlan0) |

Set the firmware's `RECEIVER_IP` to one of those addresses and keep
`UDP_PORT=12345`. Any node accepts the packet even if the pod is scheduled
elsewhere. Prefer a node on the same L2/Wi-Fi network as the ESP32; the tested
path is a Wi-Fi-connected node such as `192.168.1.50`.

Test the full path from another machine with a known 0.2–12 s mono PCM16/16 kHz
WAV:

```sh
.venv/bin/python udp-receiver/tools/send_wav.py command.wav --host 192.168.1.50
```

If you get no ACK, check in this order: the token matches, the Service is
`open-lexa-udp`, the node firewall allows UDP 12345, and the firmware uses a
LAN address rather than a Tailscale address.

## Dedicated Hailo Whisper API

`udp-receiver/deploy` and `whisper-api/deploy` create these independent workloads:

```text
ESP32 -- UDP --> open-lexa (receiver) -- HTTP --> open-lexa-whisper (ClusterIP) --> Hailo-10H
```

- `open-lexa` owns packet reassembly, command handling and the SMB archive.
  It uses the `udp-receiver/build/Dockerfile` image and requests no Hailo resource.
- `open-lexa-whisper` owns HailoRT, the model, and one
  `hailo.ai/h10h` allocation. It exposes `POST /v1/audio/transcriptions` plus
  `/healthz` and `/readyz` on a ClusterIP-only Service.
- The API accepts mono WAV PCM16LE/16 kHz, the exact audio format produced by
  the receiver. It serializes inference on one Hailo owner thread because the
  HailoRT handle is thread-affine.

The API image (`whisper-api/build/Dockerfile`) bakes in checksum-verified
Hailo Whisper-Base and Whisper-Small HEFs. It does not download them at pod
startup; the API switches the loaded model on its owner thread between
requests. Build arguments pin URLs and SHA-256 values. The admin's local STT
model selector chooses `whisper-base` or `whisper-small`; the legacy
`whisper-1` model name aliases Base.

Check the deployed path end to end with every valid recording in
`recordings/`:

```sh
kubectl -n open-lexa logs deployment/open-lexa-whisper -c whisper-api
.venv/bin/python udp-receiver/tools/test_hailo_recordings.py \
  --expected-backend dual --whisper-deployment open-lexa-whisper
```

### Configure STT: local, cloud or dual

The default is **dual**: the receiver sends every recording to the local Hailo
Whisper API and to cloud `gpt-transcribe` in parallel. The GPT transcript is
automatically saved as the corrected transcript in the admin UI, so labeling
only needs an intent and entities. Cloud output wins, local output is the
fallback when the cloud call fails.

```yaml
# Dual (default; OPENAI_API_KEY in Secret/open-lexa-secrets)
STT_BACKEND: "dual"
STT_HTTP_URL: "http://open-lexa-whisper:8080/v1"
STT_HTTP_MODEL: "whisper-base"
OPENAI_STT_MODEL: "gpt-transcribe"
# OPENAI_BASE_URL: "https://api.openai.com/v1"
```

Single-endpoint modes remain available:

```yaml
# Local in-cluster Hailo API only
STT_BACKEND: "http"
STT_HTTP_URL: "http://open-lexa-whisper:8080/v1"
STT_HTTP_MODEL: "whisper-base"

# A cloud OpenAI-compatible endpoint only
STT_BACKEND: "http"
STT_HTTP_URL: "https://provider.example/v1"
STT_HTTP_MODEL: "provider-model-name"
# STT_HTTP_API_KEY is a key in Secret/open-lexa-secrets, never a ConfigMap.

# Direct OpenAI client only
STT_BACKEND: "openai"
OPENAI_STT_MODEL: "gpt-transcribe"
```

`local` and `openai-compatible` remain aliases for `http`. Without an explicit
`STT_BACKEND`, the receiver selects `dual` when both `STT_HTTP_URL` and
`OPENAI_API_KEY` are set, otherwise it falls back to `openai`. Any alternative
server must accept `POST /v1/audio/transcriptions` with a WAV file.

The admin UI publishes the same choice to `_config/stt.json`; its dual mode is
the default. The archive sidecar keeps both transcripts (`transcripts.local`,
`transcripts.cloud`) plus `corrected_text`/`corrected_source`.

## Security notes

- The UDP protocol has a shared token but **no encryption or replay
  protection**. Expose the LoadBalancer only on the trusted LAN; do not
  publish it to the internet.
- The dedicated Whisper API is unprivileged. The Hailo device plugin injects
  only its allocated device after the pod requests `hailo.ai/h10h: 1`; keep
  that node profile for trusted homelab workloads.
- The ServiceLB source-IP masking means the optional `--device-ip` check is
  not an access control in this deployment. Use the token.

## SMB archive (recordings + transcripts)

Use a dedicated Samba account and share so the receiver cannot touch your
other shares:

- Unix + Samba user `open-lexa` (no home, `/sbin/nologin`).
- Share `[open-lexa]` with `valid users = open-lexa`; add
  `invalid users = open-lexa` to your other shares.
- Credentials live in the Secret `open-lexa/open-lexa-smb`
  (`SMB_USERNAME`, `SMB_PASSWORD`).

`udp-receiver/deploy` enables it: the Deployment's `envFrom` includes
`secretRef: open-lexa-smb` and the ConfigMap sets (the share root is already
dedicated, so the path is empty; replace the server with yours):

```yaml
  SMB_SERVER: "nas.example.lan"
  SMB_SHARE: "open-lexa"
  SMB_PATH: ""
```

Rotate the password with:

```sh
PW=$(openssl rand -hex 24)
# On the Samba host (or: kubectl exec into your Samba container):
printf '%s\n%s\n' "$PW" "$PW" | smbpasswd -s open-lexa
kubectl -n open-lexa create secret generic open-lexa-smb --from-literal=SMB_USERNAME=open-lexa \
  --from-literal=SMB_PASSWORD="$PW" --dry-run=client -o yaml | kubectl apply -f -
unset PW
```

## Labeling admin UI

`admin-dashboard/deploy/` runs `open-lexa-admin` (see [`LABELING.md`](LABELING.md),
[`HANDLERS.md`](HANDLERS.md)):
SQLite on a `longhorn-retain` PVC, recordings read from the `open-lexa` SMB share,
`_lexicon/lexicon.json`, `_config/handlers.json` and `_dataset/labels.jsonl` exported back to it. Reachable only
through the Tailscale Ingress (`https://open-lexa-admin.<tailnet>.ts.net`) with basic auth.

```sh
# Once: admin password (user "admin"); printed only when you read it back.
kubectl -n open-lexa create secret generic open-lexa-admin \
  --from-literal=ADMIN_PASSWORD="$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-20)"
kubectl apply -k admin-dashboard/deploy
kubectl -n open-lexa get secret open-lexa-admin -o jsonpath='{.data.ADMIN_PASSWORD}' | base64 -d; echo
```

The image (`admin-dashboard/build/Dockerfile`, arm64) contains both the admin
and the receiver package; see [`BUILD.md`](BUILD.md).
