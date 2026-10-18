FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    FASTEMBED_CACHE_PATH=/app/.models \
    ENVIRONMENT=production

# Hugging Face Spaces runs containers as uid 1000
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding + reranker models into the image so cold starts don't download it
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.models'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/app/.models')"

COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser app ./app
RUN chown -R appuser:appuser /app/.models

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).status == 200 else 1)"

# Migrations must succeed before the server starts (set -e semantics via &&)
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips='*' --workers 1"]
