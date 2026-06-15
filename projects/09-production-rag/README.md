# 09-production-rag

目标：在 `08-industrial-rag` 后端能力基础上，演进成一个可部署到服务器的完整 RAG 系统。09 面向真实上线形态组织：TypeScript 前端、FastAPI 后端、入库任务、运维工具、发布门禁和 Docker Compose 部署。

## 当前状态

- 前端工作台：`frontend/`
- 后端 API：`serve.py`
- RAG 核心模块：`rag_core/`
- 来源与产物 API：`rag_core/sources.py`、`rag_core/artifacts.py`
- 入库入口：`ingest_files.py`、`ingest_markdown.py`、`ingest_tables.py`、`ingest_text.py`、`ingest_image.py`
- 检索与回答工具：`search_*.py`、`answer.py`、`answer_multimodal.py`
- 运维工具：`check_config.py`、`monitor_events.py`、`collection_stats.py`、`list_documents.py`、`delete_document.py`
- 上线门禁：`eval_retrieval.py`、`eval_answer.py`、`release_gate.py`、`benchmark_latency.py`
- 测试和样例数据：`tests/`

## 项目准备文档

- `docs/EXECUTION_PLAN.md`：09 前后端分离、TypeScript 前端、部署和阶段执行计划。
- `docs/PREPARE_ENV.md`：正式编码前的环境检查、Node/npm/Docker 准备建议。
- `docs/frontend-design/`：前端参考图、UI 设计规格和 vibecoding 准备过程。

## 本地运行

从仓库根目录激活环境：

```bash
source .venv/bin/activate
```

进入项目目录：

```bash
cd projects/09-production-rag
```

复制并填写配置：

```bash
cp .env.example .env
```

初始化 schema：

```bash
python schema.py --reset
```

启动 API：

```bash
uvicorn serve:app --host 0.0.0.0 --port 8008
```

启动前端开发服务：

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

前端默认请求同源 `/api`。本地直接连后端时，可在页面右上角设置里把 API Base URL 改为 `http://127.0.0.1:8008`。

生产构建：

```bash
cd frontend
npm run build
```

## 生产入库

生产入库必须显式指定输入，不再默认读取教学样例：

```bash
python ingest_text.py --input /path/to/source_docs.jsonl
python ingest_image.py --input /path/to/image_docs.jsonl
python ingest_files.py --input-dir /path/to/files --tenant-id team_a --acl-group engineering
python ingest_tables.py --input-dir /path/to/tables --tenant-id team_a --acl-group ops
```

容器入库任务读取环境变量：

```bash
RAG_TEXT_INPUT=/data/source_docs.jsonl
RAG_IMAGE_INPUT=/data/image_docs.jsonl
```

运行入库 profile 时需要显式提供待入库输入，避免生产任务无输入却静默完成。

## Docker Compose

```bash
docker compose up -d milvus rag-api rag-web
```

Compose 默认使用 SiliconFlow 托管的 embedding/rerank/LLM API，并默认禁用本地图像 embedding，避免轻量服务器启动时加载 torch/transformers 本地权重：

- `RAG_EMBEDDING_BACKEND=siliconflow`
- `EMBEDDING_MODEL=BAAI/bge-m3`
- `RAG_RERANK_BACKEND=siliconflow`
- `RERANK_MODEL=BAAI/bge-reranker-v2-m3`
- `RAG_IMAGE_EMBEDDING_BACKEND=none`
- `RAG_QUERY_REWRITE_BACKEND=llm`
- `RAG_ANSWER_BACKEND=llm`
- `NEW_API_URL=https://api.siliconflow.cn`
- `RAG_LLM_API_KEY` / `LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash`
- `SILICONFLOW_API_KEY`

LLM 配置示例：

```bash
RAG_LLM_BASE_URL=https://api.siliconflow.cn
RAG_LLM_API_KEY=your-siliconflow-api-key
LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
```

如需启用本地 CLIP 图像检索，再显式设置 `RAG_IMAGE_EMBEDDING_BACKEND=clip` 和 `IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32`。

访问：

```text
http://localhost:8080
```

可选入库 profile：

```bash
docker compose --profile ingest run --rm rag-ingest
```

GitHub Actions 会在 `main`/`master` 变更 09 项目时构建 GHCR 镜像。服务器如需直接拉取镜像而不是本地构建，可使用：

```bash
docker compose -f docker-compose.yml -f docker-compose.ghcr.yml pull rag-api rag-web
docker compose -f docker-compose.yml -f docker-compose.ghcr.yml up -d milvus rag-api rag-web
```

## Web 功能

- 上传 PDF、Markdown、TXT、HTML、CSV、TSV 来源。
- 选择一个或多个来源进行带引用问答。
- 对回答进行点赞/点踩反馈。
- 基于已选来源调用 LLM 生成、展示并下载思维导图 Artifact。
- 在设置里切换 API 地址、Token、Tenant 和 ACL Groups。
