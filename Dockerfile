FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/resolveiq

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts
COPY data ./data
RUN mkdir -p models \
    && python scripts/train_model.py \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser /srv/resolveiq
USER appuser

EXPOSE 8000 8501
# Default: the API. docker-compose overrides the command for the UI.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
