# 09-production-rag

目标：在 `08-industrial-rag` 后端能力基础上，演进成一个可部署到服务器的完整 RAG 系统。09 不再以教学 walkthrough 为主，而是面向真实上线形态组织：后端 API、入库任务、运维工具、发布门禁，以及后续要加入的前端页面。

## 当前状态

- 后端 API：`serve.py`
- RAG 核心模块：`rag_core/`
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

未设置对应变量时，`scripts/start_ingest.sh` 会跳过对应入库步骤。

## Docker Compose

```bash
docker compose up -d milvus rag-api
```

可选入库 profile：

```bash
docker compose --profile ingest run --rm rag-ingest
```

## 后续前端目标

下一步建议新增：

```text
frontend/
  package.json
  src/
  public/
```

并在 `docker-compose.yml` 中补充前端构建/部署服务，最终形成一个可直接上线的 Web RAG 系统。
