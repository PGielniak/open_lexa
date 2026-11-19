#!/usr/bin/env bash
# Package a component's build context into its Tekton ConfigMap (and optionally
# start the build). The cluster cannot clone the repo, so Kaniko builds from a
# tarball stored in the ConfigMap referenced by <component>/build/pipelinerun.yaml.
#
#   scripts/build-context.sh admin-dashboard              # refresh the ConfigMap only
#   scripts/build-context.sh admin-dashboard --run        # + apply pipeline.yaml and start a PipelineRun
#   scripts/build-context.sh udp-receiver --run --image ghcr.io/pgielniak/open-lexa:test
#   scripts/build-context.sh whisper-api --dry-run        # only build the tarball and show its contents
#
# The directories in the tarball are the top-level sources of the COPY lines in
# <component>/build/Dockerfile (e.g. the admin image also gets udp-receiver).
set -euo pipefail

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  echo
  echo "Options: --run  --image IMAGE  --context KUBECONTEXT (default: default)"
  echo "         --namespace NS (default: tekton-builds)  --dry-run  --keep"
  exit "${1:-0}"
}

component='' run=0 dry=0 keep=0 image=''
kube_context=default namespace=tekton-builds
while (($#)); do
  case "$1" in
    -h|--help) usage ;;
    --run) run=1 ;;
    --dry-run) dry=1 ;;
    --keep) keep=1 ;;
    --image) image=${2:?--image needs a value}; shift ;;
    --context) kube_context=${2:?--context needs a value}; shift ;;
    --namespace) namespace=${2:?--namespace needs a value}; shift ;;
    -*) echo "unknown option: $1" >&2; usage 2 ;;
    *) [[ -z $component ]] || { echo "only one component path allowed" >&2; usage 2; }
       component=$1 ;;
  esac
  shift
done
[[ -n $component ]] || usage 2

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
# Accept "admin-dashboard", "./admin-dashboard/", an absolute path or any file inside it.
target=$(realpath "$component")
[[ -d $target ]] || target=$(dirname "$target")
rel=${target#"$repo"/}
[[ $rel != "$target" ]] || { echo "$component is not inside $repo" >&2; exit 1; }
rel=${rel%%/*}

dockerfile=$repo/$rel/build/Dockerfile
pipeline=$repo/$rel/build/pipeline.yaml
pipelinerun=$repo/$rel/build/pipelinerun.yaml
for f in "$dockerfile" "$pipeline" "$pipelinerun"; do
  [[ -f $f ]] || { echo "missing $f" >&2; exit 1; }
done

configmap=$(awk '/configMap:/ {getline; if ($1 == "name:") {print $2; exit}}' "$pipelinerun")
[[ -n $configmap ]] || { echo "no source configMap in $pipelinerun" >&2; exit 1; }

# Top-level directories copied by the Dockerfile (skip --from=<stage> copies).
mapfile -t dirs < <(awk 'toupper($1) ~ /^(COPY|ADD)$/ {
    for (i = 2; i < NF; i++) if ($i !~ /^--/) { split($i, p, "/"); print p[1] }
  }' "$dockerfile" | sort -u)
sources=()
for d in "${dirs[@]}"; do
  [[ -e $repo/$d ]] && sources+=("$d")
done
((${#sources[@]})) || { echo "no COPY sources found in $dockerfile" >&2; exit 1; }
[[ -f $repo/.dockerignore ]] && sources=(.dockerignore "${sources[@]}")

tmpdir=$(mktemp -d)
archive=$tmpdir/$configmap.tar.gz
((keep)) || trap 'rm -rf "$tmpdir"' EXIT

tar -C "$repo" --exclude=__pycache__ --exclude='*.egg-info' --exclude=.pytest_cache \
  --exclude=tests --exclude=deploy --exclude=tools --exclude=.venv \
  -czf "$archive" "${sources[@]}"

size=$(stat -c %s "$archive")
echo "component:  $rel"
echo "sources:    ${sources[*]}"
echo "archive:    $archive ($((size / 1024)) KiB)"
echo "configmap:  $namespace/$configmap (context $kube_context)"

# The API server caps a ConfigMap's (decoded) data at 1 MiB.
if ((size > 1048576)); then
  echo "archive too large for a ConfigMap (1 MiB limit)" >&2
  exit 1
fi

if ((dry)); then
  tar -tzf "$archive" | grep -v '/$'
  ((keep)) && echo "kept $archive"
  exit 0
fi

kc=(kubectl --context "$kube_context")
# Server-side apply: client-side apply would also copy the archive into the
# last-applied annotation, which is limited to 256 KiB.
"${kc[@]}" -n "$namespace" create configmap "$configmap" \
  --from-file=context.tar.gz="$archive" --dry-run=client -o yaml \
  | "${kc[@]}" apply --server-side --force-conflicts --field-manager=build-context -f -

if ((run)); then
  "${kc[@]}" apply -f "$pipeline"
  if [[ -n $image ]]; then
    # Replace the value that follows "- name: image" in the params list.
    sed "/- name: image/{n;s|value: .*|value: $image|}" "$pipelinerun" \
      | "${kc[@]}" -n "$namespace" create -f -
  else
    "${kc[@]}" -n "$namespace" create -f "$pipelinerun"
  fi
  echo "follow: ${kc[*]} -n $namespace get pipelinerun -l app.kubernetes.io/name=open-lexa --sort-by=.metadata.creationTimestamp"
else
  echo "start the build: ${kc[*]} -n $namespace create -f $rel/build/pipelinerun.yaml"
fi
