FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY setup.py .
COPY . ./src/serving/

RUN pip install --upgrade pip && \
    pip install -r src/serving/requirements.txt
    # pip install -e .

COPY .env ./src/serving/.env

# EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "-m", "src.serving.model_run"]
