# Dedicated Hailo Whisper transcription API.  The UDP receiver deliberately
# does not use this image or request a Hailo device. Build from the repository
# root on ARM64:
#   docker build -f whisper-api/build/Dockerfile .
FROM ghcr.io/pgielniak/hailort@sha256:40b78877754c4d04aa108b550dd52feb44dfea207da88cae3210ccfdcb2fd8b3

# These values describe the HailoRT 5.4.0 Whisper HEFs bundled in the image.
# Both Base and Small ship together so the admin portal can switch models at
# runtime (the API reloads the requested HEF between transcriptions). Override
# the URL/checksum/file args together to test different compatible models.
ARG HAILO_WHISPER_BASE_URL="https://dev-public.hailo.ai/v5.4.0/blob/Whisper-Base.hef"
ARG HAILO_WHISPER_BASE_SHA256="d85e0ba12eb8c46d6f6a7dd49125ef8497f5959289a5d1e23cba88e1d9ace372"
ARG HAILO_WHISPER_BASE_FILE="Whisper-Base.hef"
ARG HAILO_WHISPER_SMALL_URL="https://dev-public.hailo.ai/v5.4.0/blob/Whisper-Small.hef"
ARG HAILO_WHISPER_SMALL_SHA256="48161d62f788329a200eb6a94f6a241623f49267f20d749f5c3495949a8b4470"
ARG HAILO_WHISPER_SMALL_FILE="Whisper-Small.hef"

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
       "$HAILO_WHISPER_BASE_URL" -o "/models/$HAILO_WHISPER_BASE_FILE" \
    && echo "$HAILO_WHISPER_BASE_SHA256  /models/$HAILO_WHISPER_BASE_FILE" | sha256sum -c - \
    && chmod 0444 "/models/$HAILO_WHISPER_BASE_FILE" \
    && curl --fail --location --retry 3 --retry-delay 2 \
       "$HAILO_WHISPER_SMALL_URL" -o "/models/$HAILO_WHISPER_SMALL_FILE" \
    && echo "$HAILO_WHISPER_SMALL_SHA256  /models/$HAILO_WHISPER_SMALL_FILE" | sha256sum -c - \
    && chmod 0444 "/models/$HAILO_WHISPER_SMALL_FILE"

RUN groupadd --gid 10001 openlexa \
    && useradd --uid 10001 --gid openlexa --no-create-home \
       --home-dir /app --shell /usr/sbin/nologin openlexa

ENV PATH="/opt/venv/bin:$PATH"

USER 10001:10001

EXPOSE 8080/tcp

ENTRYPOINT ["open-lexa-whisper-api"]
