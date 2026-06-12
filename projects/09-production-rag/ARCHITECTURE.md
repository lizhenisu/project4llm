# 09-production-rag Architecture

09 的定位是可部署的 RAG Web 系统。当前阶段先保留 08 已完成的后端能力，并把教学资产移入 `tests/`，为后续加入前端和服务器部署结构留出清晰边界。

## Runtime Shape

```text
client / frontend
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
├── scripts/                  # container entrypoints
├── tests/
│   ├── fixtures/data/        # sample and eval fixtures
│   ├── smoke/                # smoke checks moved out of production root
│   └── walkthrough/          # legacy walkthroughs kept as test/demo assets
├── Dockerfile
├── docker-compose.yml
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

## Next Frontend Step

The next structural addition should be:

```text
frontend/
├── package.json
├── src/
└── public/
```

The frontend should call the existing FastAPI endpoints first, then Docker Compose can add a web service or static reverse-proxy deployment target.
