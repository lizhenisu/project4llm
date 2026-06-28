FROM python:3.14-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY projects/09-production-rag/requirements-api.txt /app/projects/09-production-rag/requirements-api.txt
RUN pip install --no-cache-dir \
    --index-url "${PIP_INDEX_URL}" \
    -r /app/projects/09-production-rag/requirements-api.txt

COPY projects/09-production-rag /app/projects/09-production-rag

WORKDIR /app/projects/09-production-rag

EXPOSE 8008

CMD ["./scripts/start_api.sh"]
