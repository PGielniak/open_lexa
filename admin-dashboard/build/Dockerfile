# Open Lexa labeling admin UI (FastAPI). Reads recordings from the SMB archive,
# keeps labels in SQLite on its own volume and publishes the lexicon to SMB.
# Needs the receiver package too; build from the repository root:
#   docker build -f admin-dashboard/build/Dockerfile .
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY udp-receiver ./udp-receiver
COPY admin-dashboard ./admin-dashboard

RUN pip install ./udp-receiver ./admin-dashboard \
    && groupadd --gid 10001 openlexa \
    && useradd --uid 10001 --gid openlexa --no-create-home \
       --home-dir /app --shell /usr/sbin/nologin openlexa \
    && install -d -o 10001 -g 10001 /data

USER 10001:10001

EXPOSE 8000/tcp

ENTRYPOINT ["open-lexa-admin"]
