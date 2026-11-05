# Building the images

Every container component has a `build/` directory with its `Dockerfile` and a
self-contained Tekton `pipeline.yaml` + `pipelinerun.yaml`:

| Component | Dockerfile | Image | Context ConfigMap |
| --- | --- | --- | --- |
| UDP receiver | `udp-receiver/build/Dockerfile` | `ghcr.io/pgielniak/open-lexa` | `open-lexa-receiver-context` |
| Whisper API | `whisper-api/build/Dockerfile` | `ghcr.io/pgielniak/open-lexa-whisper-hailo` | `open-lexa-whisper-api-context` |
| Admin dashboard | `admin-dashboard/build/Dockerfile` | `ghcr.io/pgielniak/open-lexa-admin` | `open-lexa-admin-context` |

The build context is always the **repository root**, because the admin image
installs the receiver package too. The root `.dockerignore` keeps tests,
manifests, firmware and recordings out of the context.

```sh
docker build -f udp-receiver/build/Dockerfile -t open-lexa:dev .
docker build -f admin-dashboard/build/Dockerfile -t open-lexa-admin:dev .
# ARM64 only: the HailoRT base image and the baked HEF are arm64.
docker buildx build --platform linux/arm64 -f whisper-api/build/Dockerfile -t open-lexa-whisper:dev .
```

The firmware is not an image; build it with `firmware/build/arduino.sh` (see
the root README).

## Tekton

Each Pipeline embeds its Kaniko task, so a component needs nothing from the
other directories. The workspace is not a git checkout: the context is a
tarball in a ConfigMap, unpacked into an emptyDir and built by Kaniko on an
ARM64 node. Example for the receiver (the other components differ only in the
directories in the tarball, listed at the top of each `pipeline.yaml`):

```sh
tar --exclude=__pycache__ --exclude='*.egg-info' --exclude=tests \
  -czf /tmp/open-lexa-receiver-context.tar.gz .dockerignore udp-receiver
kubectl --context default -n tekton-builds create configmap open-lexa-receiver-context \
  --from-file=context.tar.gz=/tmp/open-lexa-receiver-context.tar.gz \
  --dry-run=client -o yaml | kubectl --context default apply -f -

# Once, or after editing it:
kubectl --context default apply -f udp-receiver/build/pipeline.yaml

# Every build (generateName gives each run a unique name):
kubectl --context default -n tekton-builds create -f udp-receiver/build/pipelinerun.yaml
```

The admin tarball needs `.dockerignore udp-receiver admin-dashboard`; the
Whisper API tarball needs `.dockerignore whisper-api`.

Watch the run in the Dashboard or with:

```sh
kubectl --context default -n tekton-builds get pipelinerun -l app.kubernetes.io/name=open-lexa
kubectl --context default -n tekton-builds logs -f \
  --selector=tekton.dev/pipelineRun=<run-name> --all-containers --prefix
```

The push destination is `params.image` in each `pipelinerun.yaml`; bump the
tag there for a new release. The pushed digest is the run result
`IMAGE_DIGEST`; pin it (with the tag) in `<component>/deploy/kustomization.yaml`.
Reference runs of the currently pinned images:
`open-lexa-receiver-build-sltwg`, `open-lexa-whisper-api-build-2w7wj`,
`open-lexa-admin-build-ngwsh` (source revision `162d4ca`).

The build requires:

- an ARM64 node (`nodeSelector` is set in each run),
- Secret `tekton-builds/tekton-kaniko-push` with GHCR push credentials,
- network access to GHCR and, for the Whisper API, `dev-public.hailo.ai`.
