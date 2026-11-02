# Open Lexa UDP receiver: OpenAI, OpenAI-compatible HTTP (e.g. the Hailo
# Whisper API in whisper-api/), or dry-run. Build from the repository root:
#   docker build -f udp-receiver/build/Dockerfile .
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY udp-receiver ./udp-receiver

RUN pip install ./udp-receiver \
    && groupadd --gid 10001 openlexa \
    && useradd --uid 10001 --gid openlexa --no-create-home \
       --home-dir /app --shell /usr/sbin/nologin openlexa \
    && mkdir -p /recordings \
    && chown 10001:10001 /recordings

USER 10001:10001

EXPOSE 12345/udp
EXPOSE 8080/tcp

ENTRYPOINT ["open-lexa"]
