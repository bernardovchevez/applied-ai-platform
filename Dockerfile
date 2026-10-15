FROM python:3.12-slim AS base

ARG APP_ENV=production
ENV APP_ENV=${APP_ENV} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/rag:/app/platform:/app/agents" \
    PORT=8000 \
    WEB_CONCURRENCY=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY stores/ ./stores/
COPY guardrails/ ./guardrails/
COPY rag/ ./rag/
COPY agents/ ./agents/
COPY platform/ ./platform/
COPY data/ ./data/
COPY scripts/ ./scripts/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["./platform/bff/run.sh"]
