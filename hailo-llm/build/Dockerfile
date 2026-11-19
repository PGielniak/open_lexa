# hailo-ollama (HailoRT 5.5.0 GenAI) serving Gemma4-E2B on a Hailo-10H.
# Build from the repository root on ARM64:
#   docker build -f hailo-llm/build/Dockerfile .
# The licensed binary and the model manifests are NOT in git: copy them from a
# node that has `hailo-gen-ai-model-zoo` installed (see hailo-llm/README.md).
# The ~3.3 GB model is deliberately not baked in: it is pulled once into the
# hostPath cache mounted at $HOME/.local/share/hailo-ollama.
# ghcr.io/pgielniak/hailort:5.5.0
FROM ghcr.io/pgielniak/hailort@sha256:0959c46aa58ca222e70653027a38b29e3afe70e8641a1db23ac2b2369889ce96

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
       libusb-1.0-0 libudev1 libcap2 libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY hailo-llm/vendor/hailo-ollama /usr/bin/hailo-ollama
COPY hailo-llm/vendor/share/hailo-ollama /usr/share/hailo-ollama
RUN chmod 0755 /usr/bin/hailo-ollama \
    && ! ldd /usr/bin/hailo-ollama | grep 'not found' \
    && install -d -m 0777 /data

# Tool-call adapter sidecar (stdlib only; run with `python3 /opt/adapter/...`).
COPY hailo-llm/adapter /opt/adapter

# uid 1000 owns the model cache on raspberry02 (patryk).
ENV HOME=/data \
    OLLAMA_HOST=0.0.0.0:8000 \
    HAILORT_LOGGER_PATH=/tmp/hailort.log

USER 1000:1000
EXPOSE 8000/tcp
ENTRYPOINT ["hailo-ollama"]
