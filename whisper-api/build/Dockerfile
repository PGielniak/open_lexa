# Dedicated Hailo Whisper transcription API.  The UDP receiver deliberately
# does not use this image or request a Hailo device. Build from the repository
# root on ARM64:
#   docker build -f whisper-api/build/Dockerfile .
# ghcr.io/pgielniak/hailort:5.5.0
FROM ghcr.io/pgielniak/hailort@sha256:0959c46aa58ca222e70653027a38b29e3afe70e8641a1db23ac2b2369889ce96

# These values describe the HailoRT 5.5.0 Whisper HEFs bundled in the image.
# Only Whisper-Medium (new in 5.5.0, ~1.4 GB) is baked in: Base/Small were dropped
# to keep the image small (raspberry01/02 have ~1-7 GB free). Re-add an ARG +
# curl block here and the model in hailo.py/configmap to bring one back.
# Previously the admin portal could switch models at
# runtime (the API reloads the requested HEF between transcriptions). Override
# the URL/checksum/file args together to test different compatible models.
ARG HAILO_WHISPER_MEDIUM_URL="https://dev-public.hailo.ai/v5.5.0/blob/Whisper-Medium.hef"
ARG HAILO_WHISPER_MEDIUM_SHA256="64423764f8138995eb51ef64478ead2ddebeffdf3eec50fe68b8abca0f1613de"
ARG HAILO_WHISPER_MEDIUM_FILE="Whisper-Medium.hef"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    HAILORT_LOGGER_PATH=/tmp/hailort.log

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY whisper-api /app/whisper-api

RUN python3 -m venv --system-site-packages /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir /app/whisper-api \
    && /opt/venv/bin/python -c 'from hailo_platform.genai import Speech2Text, Speech2TextTask'

RUN install -d -m 0755 /models \
    && curl --fail --location --retry 3 --retry-delay 2 \
       "$HAILO_WHISPER_MEDIUM_URL" -o "/models/$HAILO_WHISPER_MEDIUM_FILE" \
    && echo "$HAILO_WHISPER_MEDIUM_SHA256  /models/$HAILO_WHISPER_MEDIUM_FILE" | sha256sum -c - \
    && chmod 0444 "/models/$HAILO_WHISPER_MEDIUM_FILE"

RUN groupadd --gid 10001 openlexa \
    && useradd --uid 10001 --gid openlexa --no-create-home \
       --home-dir /app --shell /usr/sbin/nologin openlexa

ENV PATH="/opt/venv/bin:$PATH"

USER 10001:10001

EXPOSE 8080/tcp

ENTRYPOINT ["open-lexa-whisper-api"]
