# Dedicated Hailo Whisper transcription API.  The UDP receiver deliberately
# does not use this image or request a Hailo device. Build from the repository
# root on ARM64:
#   docker build -f whisper-api/build/Dockerfile .
# ghcr.io/pgielniak/hailort:5.5.0
FROM ghcr.io/pgielniak/hailort@sha256:0959c46aa58ca222e70653027a38b29e3afe70e8641a1db23ac2b2369889ce96

# These values describe the HailoRT 5.5.0 Whisper HEFs bundled in the image.
# Base, Small and Medium (new in 5.5.0, ~1.4 GB) ship together so the admin portal can switch models at
# runtime (the API reloads the requested HEF between transcriptions). Override
# the URL/checksum/file args together to test different compatible models.
ARG HAILO_WHISPER_BASE_URL="https://dev-public.hailo.ai/v5.5.0/blob/Whisper-Base.hef"
ARG HAILO_WHISPER_BASE_SHA256="90d8fbf2830f14230a6a4d9d73bcf6a43cb9b39c8c850eb28fb73331476403a7"
ARG HAILO_WHISPER_BASE_FILE="Whisper-Base.hef"
ARG HAILO_WHISPER_SMALL_URL="https://dev-public.hailo.ai/v5.5.0/blob/Whisper-Small.hef"
ARG HAILO_WHISPER_SMALL_SHA256="0a2db66e8d496f9520a3caec43696629b7b3d0c41f33dc0fcd03ff9fa2746d33"
ARG HAILO_WHISPER_SMALL_FILE="Whisper-Small.hef"
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
       "$HAILO_WHISPER_BASE_URL" -o "/models/$HAILO_WHISPER_BASE_FILE" \
    && echo "$HAILO_WHISPER_BASE_SHA256  /models/$HAILO_WHISPER_BASE_FILE" | sha256sum -c - \
    && chmod 0444 "/models/$HAILO_WHISPER_BASE_FILE" \
    && curl --fail --location --retry 3 --retry-delay 2 \
       "$HAILO_WHISPER_SMALL_URL" -o "/models/$HAILO_WHISPER_SMALL_FILE" \
    && echo "$HAILO_WHISPER_SMALL_SHA256  /models/$HAILO_WHISPER_SMALL_FILE" | sha256sum -c - \
    && chmod 0444 "/models/$HAILO_WHISPER_SMALL_FILE" \
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
