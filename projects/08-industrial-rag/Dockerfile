FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md /app/
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

COPY projects/08-industrial-rag /app/projects/08-industrial-rag

WORKDIR /app/projects/08-industrial-rag

EXPOSE 8008

CMD ["./scripts/start_api.sh"]

