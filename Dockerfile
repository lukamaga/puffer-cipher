# syntax=docker/dockerfile:1

# ---- dashboard build ---------------------------------------------------------
FROM node:22-slim AS dashboard
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

# ---- runtime -----------------------------------------------------------------
FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PUFFER_HOST=0.0.0.0 \
    PUFFER_PORT=8000 \
    PUFFER_DATA_DIR=/data \
    PUFFER_STATIC_DIR=/app/dashboard/dist

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 puffer \
    && mkdir -p /data && chown puffer:puffer /data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY --from=dashboard /app/dashboard/dist ./dashboard/dist

USER puffer
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4); sys.exit(0)"
CMD ["puffer", "serve"]
