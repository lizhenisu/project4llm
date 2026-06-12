# 09-production-rag Architecture

09 的定位是可部署的 RAG Web 系统。当前结构保留 08 已完成的后端能力，并新增独立 TypeScript 前端、来源管理 API、Studio Artifact API 和服务器部署编排。

## Runtime Shape

```text
browser
  -> rag-web: Nginx static frontend
    -> /api proxy
      -> rag-api: FastAPI service in serve.py
    -> rag_core pipeline
      -> embedding / sparse retrieval / Milvus
      -> rerank
      -> context packing
      -> LLM gateway
```

## Directory Layout

```text
projects/09-production-rag/
├── rag_core/                 # RAG domain modules
│   ├── sources.py            # upload/list/delete source lifecycle
│   └── artifacts.py          # Studio output persistence
├── frontend/                 # Vite + React + TypeScript web app
│   ├── src/
│   ├── Dockerfile
│   └── nginx.conf
├── scripts/                  # container entrypoints
├── tests/
│   ├── fixtures/data/        # sample and eval fixtures
│   ├── smoke/                # smoke checks moved out of production root
│   └── walkthrough/          # legacy walkthroughs kept as test/demo assets
├── Dockerfile
├── docker-compose.yml        # milvus + api + web
├── serve.py
├── ingest_*.py
├── search_*.py
├── answer*.py
├── eval_*.py
└── release_gate.py
```

## Production Boundary

- Runtime data is generated under `runtime/`, `object_store/`, `volumes/`, or the configured external services.
- Sample and eval data live under `tests/fixtures/data/`.
- `ingest_text.py` and `ingest_image.py` require explicit `--input`; they no longer default to fixtures.
- `scripts/start_ingest.sh` reads `RAG_TEXT_INPUT` and `RAG_IMAGE_INPUT` when running in containers.
- Uploaded sources are saved under `object_store/uploads/<tenant>/<upload-id>/` before ingestion.
- Studio artifacts are saved under `object_store/artifacts/<tenant>/`.
- `rag-web` serves static frontend assets and proxies `/api/*` to `rag-api:8008`.

## Deploy Shape

```bash
docker compose up -d milvus rag-api rag-web
```

The browser enters through `http://server:8080`. In a real server deployment, place TLS and the public domain in front of `rag-web`, then keep `rag-api`, Milvus, etcd and MinIO on the private Docker network.
