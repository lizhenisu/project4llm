# 08-industrial-rag

目标：设计一个带教学目的、但按可上线标准组织的工业级 RAG 向量检索系统。向量库使用 Milvus，文本 embedding 和 rerank 使用 BGE 家族，并预留图片等多模态 RAG 能力。

本项目不是“最小 demo”。它要回答的是：如果要把 RAG 做到生产环境，schema、metadata、hybrid search、rerank、权限、评估、部署和运维应该怎么设计。

## 0.1 学习主线

如果你当前目标是“先学会 RAG”，不要一上来就看部署、鉴权、监控。

先只看这一条主线：

0. `ARCHITECTURE.md`：先看项目分层、模块关系和从零复现顺序，建立宏观代码框架。
1. `walkthrough_core_rag.py`：先一口气看完整条链路发生了什么。
2. `schema.py`：Milvus 里为什么要显式 schema，而不是随手塞 JSON。
3. `ingest_text.py` + `rag_core/text_utils.py`：文档怎么切 chunk，chunk 里保留哪些 metadata。
4. `search_dense.py` / `search_sparse.py` / `search_hybrid.py`：dense、sparse、hybrid 各解决什么问题。
5. `rerank.py`：为什么召回以后还要二阶段排序。
6. `answer.py` + `rag_core/context.py`：证据怎么选、怎么拼 prompt、什么时候该拒答。
7. `eval_retrieval.py` / `eval_answer.py`：怎么评估“检索好不好”“回答靠不靠谱”。

建议先按这个顺序跑：

```bash
python projects/08-industrial-rag/walkthrough_core_rag.py
python projects/08-industrial-rag/schema.py --reset --explain
python projects/08-industrial-rag/ingest_text.py --explain
python projects/08-industrial-rag/search_dense.py "RAG 检索变慢时应该排查哪些环节？" --acl-group support --explain
python projects/08-industrial-rag/search_sparse.py "RAG 检索变慢时应该排查哪些环节？" --acl-group support --explain
python projects/08-industrial-rag/search_hybrid.py "RAG 检索变慢时应该排查哪些环节？" --acl-group support --explain
python projects/08-industrial-rag/rerank.py "RAG 检索变慢时应该排查哪些环节？" --acl-group support --show-candidates
python projects/08-industrial-rag/answer.py "RAG 检索变慢时应该排查哪些环节？" --acl-group support --show-trace --show-prompt-chars 400
python projects/08-industrial-rag/eval_retrieval.py
python projects/08-industrial-rag/eval_answer.py
```

学完这条主线，再进入第二层：

- 多模态 RAG：`walkthrough_multimodal_rag.py`、`ingest_image.py`、`search_multimodal.py`、`answer_multimodal.py`
- 工业化护栏：`serve.py`、`smoke_*`、`monitor_events.py`、`release_gate.py`
- 部署专题：`docker-compose.yml`、`Dockerfile`、`smoke_deploy.py`

## 0.2 V1 / V2 定义

为了避免“v2 到底指什么”这种歧义，当前 README 里的版本语义明确写成：

- `V1`：文本工业 RAG。
  包含 schema、chunk、dense/sparse/hybrid、rerank、answer、eval、serve、release gate。
- `V2`：多模态工业 RAG。
  在 V1 基础上加入 OCR/caption、`image_dense_vector`、multimodal retrieval、multimodal answer、multimodal eval、multimodal monitoring/release gate。

当前仓库状态：

- `V1`：已完成，可教学，也具备上线所需主链路。
- `V2`：主链路已完成，已经有 `search_multimodal.py`、`answer_multimodal.py`、`eval_* --mode multimodal` 和 API/monitoring/release gate 接入。
- `V2` 尚未专门落地的项：BGE visualized / VISTA 这类专用视觉 embedding backend；当前 `image_dense_vector` 走的是 CLIP backend。

## 0. 当前实现状态

当前目录已经包含一条可运行的工业 RAG 骨架：

| 文件 | 作用 |
| --- | --- |
| `schema.py` | 显式创建 Milvus collection、dense/sparse 向量索引和 metadata 字段 |
| `ingest_text.py` | 读取 JSONL 文本文档，chunk、embedding、sparse 特征并 upsert 到 Milvus |
| `ingest_markdown.py` | 读取 Markdown 目录，生成文档 metadata 并入库 |
| `ingest_files.py` | 读取 PDF、HTML、Markdown、TXT 目录，统一转换为 `SourceDocument` 后入库 |
| `ingest_tables.py` | 读取 CSV/TSV 表格，转 compact markdown table 并保留列、行范围等 metadata 后入库 |
| `ingest_image.py` | 读取图片 OCR/caption 元数据，写入文本向量和图片向量字段 |
| `rebuild_from_object_store.py` | 从归档的 canonical 文档重建 Milvus 索引 |
| `delete_document.py` | 按 `tenant_id/doc_id/doc_version` 删除文档 chunk |
| `list_documents.py` | 按租户查看已发布文档版本、chunk 数和 ACL |
| `collection_stats.py` | 输出 collection 行数、文档数、租户数和 source 类型分布 |
| `walkthrough_core_rag.py` | 用临时 Milvus Lite 串起 schema、chunk、dense/BM25/hybrid、rerank、prompt、answer、eval 的教学 walkthrough |
| `walkthrough_multimodal_rag.py` | 用临时 Milvus Lite 串起 OCR/caption、image vector、multimodal retrieval、answer、eval 的教学 walkthrough |
| `search_dense.py` | metadata filter + dense search |
| `search_sparse.py` | metadata filter + Milvus BM25 search，便于关键词召回 ablation |
| `search_hybrid.py` | Milvus hybrid search，融合 dense 向量和 Milvus BM25 |
| `search_image.py` | 使用 `image_dense_vector` 做图片向量检索 |
| `search_multimodal.py` | 融合 OCR/caption 文本 hybrid 检索和 `image_dense_vector` 图片检索 |
| `rerank.py` | 对 hybrid 候选做二阶段 rerank |
| `diagnose_retrieval.py` | 展开 dense/sparse/hybrid/rerank 的候选 rank 和 score，排查召回与 rerank 问题 |
| `diagnose_context.py` | 展开 context packing 的逐候选选择/丢弃原因，排查 prompt 证据构造 |
| `sweep_chunking.py` | 用临时 collection 对多组 chunk_size/overlap 做检索指标和延迟 sweep |
| `answer.py` | 检索、rerank、组 prompt，并通过 NewAPI 生成答案 |
| `answer_multimodal.py` | 多模态检索、context packing、图片 OCR/caption prompt 和答案生成 |
| `eval_retrieval.py` | 输出 recall、MRR、nDCG、latency 和权限泄露检查 |
| `eval_answer.py` | 输出 citation accuracy、evidence hit rate、refusal quality |
| `build_eval_from_feedback.py` | 把 runtime feedback 与检索/回答事件合并，导出可回放 eval JSONL |
| `release_gate.py` | 按上线阈值检查 retrieval/answer 指标，不达标时非零退出 |
| `benchmark_latency.py` | 回放真实 answer pipeline，汇总 rewrite/search/rerank/context/answer 分段延迟 |
| `smoke_benchmark.py` | 验证 benchmark 会走真实 text / multimodal answer pipeline，并输出对应 stage latency |
| `monitor_events.py` | 汇总 runtime 事件，输出 p50/p95/p99、检索模式、context 命中和反馈分布 |
| `scan_pii.py` | 扫描 JSONL 知识源中的邮箱、手机号、身份证号、API key 形态 |
| `smoke_security.py` | 验证 PII 工具和 tenant/ACL 防泄露行为 |
| `smoke_event_redaction.py` | 验证 runtime 事件会脱敏 query、feedback 和证据 preview 中的 PII |
| `smoke_auth_context.py` | 验证 API 从服务端请求头读取 tenant/ACL，并拒绝缺失 token/context |
| `smoke_context.py` | 验证 context packing 和低置信拒答 |
| `smoke_context_diagnosis.py` | 验证 context packing 诊断能输出逐候选 drop/select 原因 |
| `smoke_context_backfill.py` | 验证后续候选会在 context packing 时补位，不会因提前截断丢失 |
| `smoke_rewrite.py` | 验证 query rewrite 和 trace 记录 |
| `smoke_answer_eval.py` | 验证 citation 解析和拒答识别 |
| `smoke_answer_quality_eval.py` | 验证 answer correctness 和 faithfulness 规则评估 |
| `smoke_file_ingest.py` | 验证 HTML/TXT 文件解析、chunk、入库和检索 |
| `smoke_table_ingest.py` | 验证 CSV 表格转 markdown table、metadata、chunk、入库和检索 |
| `smoke_chunk_structure.py` | 验证 chunk 时不会拆断 markdown table 和 fenced code block |
| `smoke_chunk_sweep.py` | 验证 chunk 参数 sweep 能输出 chunk 数、召回和延迟 |
| `smoke_search_params.py` | 验证 HNSW `ef` 和 sparse drop ratio 等检索调参会传入 Milvus search |
| `smoke_multimodal_search.py` | 验证图片 OCR/caption 文本通道和 image vector 通道会被融合召回 |
| `smoke_multimodal_prompt.py` | 验证图片证据进入 prompt 时保留 image URI、bbox、linked doc 和 OCR/caption 限制提示 |
| `smoke_multimodal_eval.py` | 验证 `eval_retrieval.py --mode multimodal` 可评估图文检索 recall/MRR/latency |
| `smoke_multimodal_answer.py` | 验证多模态检索证据可进入回答链路并生成 citation answer |
| `smoke_multimodal_answer_eval.py` | 验证 `eval_answer.py --mode multimodal` 可评估图片证据回答质量 |
| `smoke_pdf_page_metadata.py` | 验证 PDF 按页入库，并在 metadata/prompt 中保留页码 citation 信息 |
| `smoke_heading_metadata.py` | 验证 Markdown/HTML heading path 会进入 metadata 和 chunk 标题路径 |
| `smoke_feedback_eval_export.py` | 验证线上 feedback 事件可导出为离线 eval set |
| `smoke_eval_filters.py` | 验证 eval 样本中的 ACL、版本、source type 和 chunk 级期望会被回放 |
| `smoke_sparse_ablation.py` | 验证 sparse-only 检索和 `eval_retrieval.py --mode sparse` |
| `smoke_retrieval_diagnosis.py` | 验证检索诊断工具能输出 rank/score 并导出 JSONL |
| `smoke_lifecycle.py` | 验证 `doc_version` 版本过滤和不存在版本不返回 |
| `smoke_current_version.py` | 验证默认查询只检索 current-version registry 中的发布版本 |
| `smoke_current_version_unpublish.py` | 验证删除当前版本时会从 current-version registry 取消发布 |
| `smoke_embedding_model_filter.py` | 验证查询不会混用旧 embedding 模型写入的向量 |
| `smoke_source_filter.py` | 验证 query/API 的 `source_types` metadata filter |
| `smoke_object_store_rebuild.py` | 验证 canonical text 归档后可重建 Milvus 索引 |
| `smoke_object_store_delete_tombstone.py` | 验证删除 tombstone 会阻止 object store 重建时复活已删除文档 |
| `smoke_observability.py` | 验证 runtime 事件包含 raw hits、rerank hits、final context 和 LLM 延迟 |
| `smoke_monitoring.py` | 验证 runtime 事件可聚合成线上监控指标 |
| `smoke_release_gate.py` | 验证 release gate 会调用 eval、输出 report，并按阈值失败退出 |
| `smoke_container_config.py` | 验证 `.env.example`、Dockerfile 和 `docker-compose.yml` 的部署约定对齐 |
| `smoke_e2e.py` | 重建库、入库、hybrid rerank、图片检索的一键验收 |
| `smoke_api.py` | 直接调用 FastAPI app，验证 `/health`、`/search`、`/query`、`/feedback` |
| `smoke_api_multimodal.py` | 直接调用 FastAPI app，验证 `/search`、`/query` 的多模态分支 |
| `smoke_readiness.py` | 验证 `/ready` 会检查 Milvus collection、schema 字段和向量维度 |
| `smoke_deploy.py` | 默认自起临时 HTTP API 做部署 smoke；设置 `RAG_API_URL` 时可改为验证外部已部署服务 |
| `smoke_deploy_contract.py` | 验证部署 smoke 的 auth header 和反馈 selected docs 构造 |
| `smoke_llm.py` | 使用 NewAPI 配置做 LLM 网关连通性测试 |
| `smoke_milvus.py` | 验证当前 `MILVUS_URI` 可连接且 collection 可 load |
| `smoke_models.py` | 可选加载 BGE/reranker/CLIP 后端，并在临时 Milvus 上做一次真实 ingest -> retrieve -> rerank smoke |
| `check_config.py` | 打印脱敏配置并检查 Milvus 连接 |
| `serve.py` | FastAPI `/search`、`/query`、`/feedback` 和 `/health` 服务入口 |
| `docker-compose.yml` | Milvus Standalone 最小部署参考 |
| `Dockerfile` | RAG API/ingest 容器镜像 |
| `.env.example` | 本地和生产配置模板，不包含真实密钥 |
| `RELEASE_CHECKLIST.md` | 从教学 smoke 到上线前验收的检查清单 |

本地默认使用 `RAG_EMBEDDING_BACKEND=bge` 和 `RAG_RERANK_BACKEND=bge`，首次使用需先下载模型（ModelScope 源，国内可直接下载）：

```bash
python projects/08-industrial-rag/download_models.py
```

生产环境配置参考：

```bash
export EMBEDDING_MODEL=BAAI/bge-m3
export RERANK_MODEL=BAAI/bge-reranker-v2-m3
export RAG_MODEL_DEVICE=cpu          # 或 cuda
export RAG_MODEL_DTYPE=fp32          # 或 fp16/bf16/auto
export RAG_EMBED_BATCH_SIZE=8
export RAG_RERANK_BATCH_SIZE=8
```

如果模型本地缓存未命中且需通过 Hugging Face Hub 下载，可以设置镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
```

`HF_ENDPOINT` 和 `HF_HUB_DISABLE_XET` 是 Hugging Face 官方支持的环境变量，用于在部分网络环境下绕开 `hf-xet` 或加快模型权重加载。

PII 策略：

```bash
export RAG_PII_POLICY=warn    # 默认：发现 PII 打印警告，仍允许入库
export RAG_PII_POLICY=redact  # 入库前脱敏
export RAG_PII_POLICY=fail    # 发现 PII 直接失败
```

Context packing 和拒答阈值：

```bash
export RAG_MAX_CONTEXT_CHARS=6000      # 兼容旧变量名；BGE 路径会按 tokenizer token 预算计数
export RAG_MAX_CHUNKS_PER_DOC=2
export RAG_MIN_RERANK_SCORE=       # 空值表示不启用最低分阈值
export RAG_OBJECT_STORE_DIR=projects/08-industrial-rag/object_store
```

Query rewrite：

```bash
export RAG_QUERY_REWRITE_BACKEND=llm        # 默认，使用 NewAPI LLM 改写
export RAG_QUERY_REWRITE_BACKEND=none       # 完全不改写
```

`llm` rewrite 需要配置 `NEW_API_URL` / `NEW_API_KEY`，未配置会直接失败。生产中应使用 LLM 或专门 query rewrite 模型把追问改写成独立检索问题。

Milvus index/search 调参：

```bash
export RAG_DENSE_HNSW_M=16
export RAG_DENSE_HNSW_EF_CONSTRUCTION=100
export RAG_DENSE_SEARCH_EF=128
export RAG_IMAGE_HNSW_M=16
export RAG_IMAGE_HNSW_EF_CONSTRUCTION=100
export RAG_IMAGE_SEARCH_EF=128
export RAG_SPARSE_DROP_RATIO_BUILD=0.2
export RAG_SPARSE_DROP_RATIO_SEARCH=0.0
```

`RAG_*_HNSW_*` 和 `RAG_SPARSE_DROP_RATIO_BUILD` 是建索引参数，修改后需要重建 collection 或重建索引才会生效；`RAG_DENSE_SEARCH_EF`、`RAG_IMAGE_SEARCH_EF` 和 `RAG_SPARSE_DROP_RATIO_SEARCH` 是查询参数，每次 search 会读取当前配置。

API auth context：

```bash
export RAG_REQUIRE_AUTH_CONTEXT=1
export RAG_API_TOKEN="dev-only-token"
```

启用后，`/search` 和 `/query` 需要：

```text
Authorization: Bearer dev-only-token
X-RAG-Tenant-ID: team_a
X-RAG-ACL-Groups: support,ops
```

生产中 `tenant_id` 和 `acl_groups` 应来自认证服务或 API 网关注入的服务端可信上下文；请求 body 里的同名字段只保留给教学兼容模式。

图片向量使用真实视觉或图文 embedding。当前 CLIP 后端配置：

```bash
export RAG_IMAGE_EMBEDDING_BACKEND=clip
export IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32
export IMAGE_EMBEDDING_DIM=512
```

如果 Hugging Face 下载受限，可以配置镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### 本地快速运行

首次使用请先下载模型（~4.5 GB，ModelScope 源）：

```bash
python projects/08-industrial-rag/download_models.py
```

从仓库根目录执行：

```bash
source .venv/bin/activate
python projects/08-industrial-rag/schema.py --reset
python projects/08-industrial-rag/check_config.py
python projects/08-industrial-rag/ingest_text.py
python projects/08-industrial-rag/ingest_files.py --input-dir notes --tenant-id team_a --acl-group engineering
python projects/08-industrial-rag/ingest_tables.py --input-dir knowledge_base --tenant-id team_a --acl-group ops
python projects/08-industrial-rag/ingest_image.py
python projects/08-industrial-rag/list_documents.py --tenant-id team_a
python projects/08-industrial-rag/collection_stats.py --tenant-id team_a
python projects/08-industrial-rag/rebuild_from_object_store.py --reset
python projects/08-industrial-rag/search_sparse.py "ECOM_7741 webhook 签名" --tenant-id team_a
python projects/08-industrial-rag/search_hybrid.py "为什么 hybrid search 要用 BM25" --tenant-id team_a
python projects/08-industrial-rag/search_image.py "RAG Dashboard latency recall" --tenant-id team_a --acl-group ops
python projects/08-industrial-rag/search_multimodal.py "RAG Dashboard latency recall" --tenant-id team_a --acl-group ops
python projects/08-industrial-rag/rerank.py "退款需要提交什么材料" --tenant-id team_a
python projects/08-industrial-rag/diagnose_retrieval.py "RAG 检索变慢时应该排查什么" --tenant-id team_a --acl-group ops
python projects/08-industrial-rag/diagnose_context.py "RAG 检索变慢时应该排查什么" --tenant-id team_a --acl-group ops --max-context-chars 1200
python projects/08-industrial-rag/sweep_chunking.py --mode hybrid --spec 400:80 --spec 700:100
python projects/08-industrial-rag/answer.py "RAG 检索变慢时应该排查什么" --tenant-id team_a
python projects/08-industrial-rag/answer_multimodal.py "RAG Dashboard latency recall" --tenant-id team_a --acl-group ops
python projects/08-industrial-rag/eval_retrieval.py --mode dense
python projects/08-industrial-rag/eval_retrieval.py --mode sparse
python projects/08-industrial-rag/eval_retrieval.py --mode hybrid
python projects/08-industrial-rag/eval_retrieval.py --mode rerank
python projects/08-industrial-rag/eval_retrieval.py --mode multimodal
python projects/08-industrial-rag/eval_answer.py
python projects/08-industrial-rag/eval_answer.py --mode multimodal --input projects/08-industrial-rag/data/multimodal_eval_queries.jsonl
python projects/08-industrial-rag/build_eval_from_feedback.py --include-negative
python projects/08-industrial-rag/release_gate.py
python projects/08-industrial-rag/benchmark_latency.py
python projects/08-industrial-rag/benchmark_latency.py --query-mode multimodal --query "RAG Dashboard latency recall" --tenant-id team_a --acl-group ops --source-type image
python projects/08-industrial-rag/monitor_events.py
python projects/08-industrial-rag/scan_pii.py projects/08-industrial-rag/data/sample_docs.jsonl projects/08-industrial-rag/data/sample_images.jsonl --fail
python projects/08-industrial-rag/smoke_e2e.py
python projects/08-industrial-rag/smoke_api.py
python projects/08-industrial-rag/smoke_api_multimodal.py
python projects/08-industrial-rag/smoke_readiness.py
python projects/08-industrial-rag/smoke_security.py
python projects/08-industrial-rag/smoke_event_redaction.py
python projects/08-industrial-rag/smoke_auth_context.py
python projects/08-industrial-rag/smoke_context.py
python projects/08-industrial-rag/smoke_context_diagnosis.py
python projects/08-industrial-rag/smoke_context_backfill.py
python projects/08-industrial-rag/smoke_rewrite.py
python projects/08-industrial-rag/smoke_answer_eval.py
python projects/08-industrial-rag/smoke_answer_quality_eval.py
python projects/08-industrial-rag/smoke_file_ingest.py
python projects/08-industrial-rag/smoke_table_ingest.py
python projects/08-industrial-rag/smoke_chunk_structure.py
python projects/08-industrial-rag/smoke_chunk_sweep.py
python projects/08-industrial-rag/smoke_search_params.py
python projects/08-industrial-rag/smoke_multimodal_search.py
python projects/08-industrial-rag/smoke_multimodal_prompt.py
python projects/08-industrial-rag/smoke_multimodal_eval.py
python projects/08-industrial-rag/smoke_multimodal_answer.py
python projects/08-industrial-rag/smoke_multimodal_answer_eval.py
python projects/08-industrial-rag/smoke_pdf_page_metadata.py
python projects/08-industrial-rag/smoke_heading_metadata.py
python projects/08-industrial-rag/smoke_feedback_eval_export.py
python projects/08-industrial-rag/smoke_eval_filters.py
python projects/08-industrial-rag/smoke_sparse_ablation.py
python projects/08-industrial-rag/smoke_retrieval_diagnosis.py
python projects/08-industrial-rag/smoke_lifecycle.py
python projects/08-industrial-rag/smoke_current_version.py
python projects/08-industrial-rag/smoke_current_version_unpublish.py
python projects/08-industrial-rag/smoke_embedding_model_filter.py
python projects/08-industrial-rag/smoke_source_filter.py
python projects/08-industrial-rag/smoke_object_store_rebuild.py
python projects/08-industrial-rag/smoke_object_store_delete_tombstone.py
python projects/08-industrial-rag/smoke_benchmark.py
python projects/08-industrial-rag/smoke_observability.py
python projects/08-industrial-rag/smoke_monitoring.py
python projects/08-industrial-rag/smoke_release_gate.py
python projects/08-industrial-rag/smoke_container_config.py
python projects/08-industrial-rag/smoke_deploy_contract.py
python projects/08-industrial-rag/smoke_deploy.py
python projects/08-industrial-rag/smoke_llm.py
python projects/08-industrial-rag/smoke_milvus.py
python projects/08-industrial-rag/smoke_models.py
```

`benchmark_latency.py` 现在直接回放 `answer.py` / `answer_multimodal.py` 的真实链路，而不是单独拼一套简化检索逻辑；text 模式会输出 `rewrite`、`embedding`、`milvus_search`、`rerank`、`context_pack`、`answer`、`total`，multimodal 模式会额外输出 `text_search`、`image_search`、`fusion` 等阶段。

也可以进入项目目录使用 Makefile：

```bash
cd projects/08-industrial-rag
make walkthrough
make walkthrough-multimodal
make schema
make ingest
make smoke
make api-smoke
make api-multimodal-smoke
make readiness-smoke
make security-smoke
make event-redaction-smoke
make auth-smoke
make context-smoke
make context-diagnosis-smoke
make context-backfill-smoke
make rewrite-smoke
make answer-eval-smoke
make answer-quality-smoke
make file-ingest-smoke
make table-ingest-smoke
make chunk-structure-smoke
make chunk-sweep-smoke
make search-param-smoke
make multimodal-search-smoke
make multimodal-prompt-smoke
make multimodal-eval-smoke
make multimodal-answer-smoke
make multimodal-answer-eval-smoke
make pdf-page-smoke
make heading-metadata-smoke
make feedback-eval-smoke
make eval-filter-smoke
make sparse-ablation-smoke
make retrieval-diagnosis-smoke
make lifecycle-smoke
make current-version-smoke
make current-version-unpublish-smoke
make embedding-model-smoke
make source-filter-smoke
make object-store-smoke
make object-store-delete-smoke
make benchmark-smoke
make observability-smoke
make monitoring-smoke
make container-config-smoke
make eval
make answer-eval
make release-gate
make benchmark
make monitor
make export-feedback-eval
make deploy-smoke
```

如果没有设置 `RAG_API_URL`，`smoke_deploy.py` / `make deploy-smoke` 会自动起一个临时本地 uvicorn 服务、灌入教学样例数据并覆盖 `/ready`、`/search`、`/query`、`/feedback`；这条本地自启链路会使用 `RAG_MILVUS_URI` 指向独立的 Milvus Lite 文件，避免和外部 `MILVUS_URI` 冲突。如果设置了 `RAG_API_URL`，则会改为验证外部已部署服务。`smoke_container_config.py` 会额外检查 `.env.example`、Dockerfile 和 `docker-compose.yml` 中的 object store/runtime 路径、MinIO root 凭据变量和 compose service 约定是否仍和文档一致。

### Docker Compose 部署

从 `projects/08-industrial-rag` 目录执行：

```bash
cp .env.example .env
docker compose up -d milvus rag-api
docker compose --profile ingest run --rm rag-ingest
RAG_API_URL=http://127.0.0.1:8008 python smoke_deploy.py
```

如果 API 启用了服务端 auth context，同时传入部署 smoke 所需的鉴权环境变量：

```bash
export RAG_REQUIRE_AUTH_CONTEXT=1
export RAG_API_TOKEN=dev-only-token
export RAG_DEPLOY_TENANT_ID=team_a
export RAG_DEPLOY_ACL_GROUPS=ops,support
RAG_API_URL=http://127.0.0.1:8008 python smoke_deploy.py
```

`rag-api` 默认连接 compose 内的 `http://milvus:19530`，并使用 BGE/CLIP/NewAPI 配置。`./object_store` 会同时挂载到 `rag-api` 和 `rag-ingest`，用于持久化 canonical 文档、删除 tombstone 和 `current_versions.json`；`./runtime` 会挂载到 `rag-api`，用于保留 retrieval/answer/feedback 事件。真实 NewAPI key 通过环境变量注入，不要写入 compose 文件。

入库 Markdown 目录：

```bash
python projects/08-industrial-rag/ingest_markdown.py \
  --input-dir notes \
  --tenant-id team_a \
  --acl-group support \
  --acl-group engineering
```

入库 PDF、HTML、Markdown、TXT 混合目录：

```bash
python projects/08-industrial-rag/ingest_files.py \
  --input-dir knowledge_base \
  --tenant-id team_a \
  --doc-version 2 \
  --acl-group support \
  --acl-group engineering
```

入库 CSV/TSV 表格目录。脚本会把每个表格转成 compact markdown table，大表按 `--rows-per-document` 拆分，并在 metadata 中保留列名、总行数、当前行范围、原始路径和格式：

```bash
python projects/08-industrial-rag/ingest_tables.py \
  --input-dir business_tables \
  --tenant-id team_a \
  --doc-version 2 \
  --rows-per-document 200 \
  --acl-group ops
```

查看当前 collection 文档版本：

```bash
python projects/08-industrial-rag/list_documents.py --tenant-id team_a
python projects/08-industrial-rag/collection_stats.py --tenant-id team_a
```

删除某篇文档的 chunk：

```bash
python projects/08-industrial-rag/delete_document.py \
  --tenant-id team_a \
  --doc-id refund-policy \
  --yes
```

默认情况下，如果删除的是当前发布版本，脚本会同步从 `current_versions.json` 中取消发布该文档，避免默认查询继续带着已删除文档的版本过滤。删除历史版本且不是当前版本时 registry 不变；确实只想清理 Milvus chunk 而不动发布状态时，传 `--keep-current-version`。

删除脚本默认还会在 object store 写入 `canonical/deleted_documents.jsonl` tombstone。`rebuild_from_object_store.py` 读取归档 canonical 文档时会跳过 tombstone 命中的文档版本，避免索引重建把已删除文档重新写回 Milvus。canonical 原文仍可用 `include_deleted=True` 的内部加载方式做审计；重新入库同一文档版本会清理对应 tombstone，相当于显式恢复。

启动 API：

```bash
source .venv/bin/activate
cd projects/08-industrial-rag
uvicorn serve:app --host 127.0.0.1 --port 8008
```

API 端点：

- `GET /health`：轻量 liveness，只返回进程是否存活。
- `GET /ready`：readiness，检查 Milvus 是否可连接、collection 是否存在、schema 关键字段和向量维度是否匹配当前配置；失败返回 503。
- `POST /search`：只返回检索和 rerank 后的证据，以及 trace；每条 hit 保留原始 `metadata`。
- `POST /query`：返回答案和 citations；每条 citation 同样保留原始 `metadata`。
- `POST /feedback`：接收用户反馈，写入 runtime feedback 事件；本地可用 `build_eval_from_feedback.py` 合并检索/回答事件，导出可回放 eval set。生产中应替换为事件表、消息队列或对象存储。

`/search` 和 `/query` 请求都支持 `history: list[str]`，用于 query rewrite；也支持 `doc_version`，用于只检索某个已发布版本；还支持 `source_types: list[str]`，用于限制 `pdf/md/html/image/api` 等来源类型。`query_mode=text|multimodal` 可切换文本链路和图文融合链路；多模态模式会走 OCR/caption text hybrid + `image_dense_vector` 融合检索。trace 会返回 `original_query`、`rewritten_query`、`rewrite_backend`、`doc_version` 和 `source_types`。多模态 hit/citation 的 `metadata` 会继续带出 `image_uri`、`bbox`、`linked_doc_id`、`fusion.channels` 等解释字段，便于前端展示和审计。

默认教学模式下，API 仍接受 body 中的 `tenant_id` 和 `acl_groups`，便于脚本和 smoke 直接调用。设置 `RAG_REQUIRE_AUTH_CONTEXT=1` 后，API 会忽略 body 中的租户和 ACL，改用 `X-RAG-Tenant-ID`、`X-RAG-ACL-Groups` 和可选 Bearer token 构造服务端 metadata filter。

如果 query 显式提到请求租户之外的 `team_xxx`，pipeline 会触发 `blocked_cross_tenant_query`，不返回任何检索上下文，避免用本租户无关证据回答跨租户问题。

当前实现会把审计事件写入 `projects/08-industrial-rag/runtime/`：

- `retrieval_events.jsonl`
- `answer_events.jsonl`
- `feedback_events.jsonl`

`retrieval_events` 和 `answer_events` 包含：

- `auth_context` 摘要。
- `trace`：query rewrite、filter、计数、context packing 丢弃原因、分段 latency。
- `raw_hits`：Milvus hybrid 原始候选摘要。
- `rerank_hits`：二阶段排序后的候选摘要。
- `final_context`：最终进入 prompt 的证据摘要。
- `llm`：回答生成模型、后端、延迟和 token usage；仅 `answer_events`。

这些运行时文件已被 `.gitignore` 忽略。生产中应替换为 Kafka、数据库事件表或对象存储。事件日志只保存短 `text_preview`，并在写入前递归脱敏邮箱、手机号、身份证号和 API key 形态；不要记录完整 API key、未脱敏隐私或长篇用户原文。

可以用 `monitor_events.py` 汇总本地 runtime 事件，得到 retrieval/answer/feedback 数量、retrieval mode 分布、请求 source type 分布、final context source type 分布、多模态 fusion channel 分布、context 命中、分段 latency 的 p50/p95/p99、LLM latency、top context docs 和反馈 rating 分布：

```bash
python projects/08-industrial-rag/monitor_events.py
```

可以把线上反馈沉淀成离线评估集。正反馈默认使用用户 `selected_doc_ids`，没有选择时退回当次 `final_context`；加 `--include-negative` 后，负反馈且没有选中文档会导出为 `answerable=false` 的拒答/坏例样本：

```bash
python projects/08-industrial-rag/build_eval_from_feedback.py \
  --include-negative \
  --output projects/08-industrial-rag/data/feedback_eval_queries.jsonl
python projects/08-industrial-rag/eval_retrieval.py \
  --input projects/08-industrial-rag/data/feedback_eval_queries.jsonl \
  --mode rerank
```

`eval_retrieval.py` 和 `eval_answer.py` 支持 `--json-output` 写出机器可读指标；`release_gate.py` 会直接运行两类评估并按阈值失败退出，适合接入 CI 或上线前 checklist：

```bash
python projects/08-industrial-rag/release_gate.py \
  --min-recall 0.90 \
  --multimodal-input projects/08-industrial-rag/data/multimodal_eval_queries.jsonl \
  --multimodal-answer-input projects/08-industrial-rag/data/multimodal_eval_queries.jsonl \
  --min-multimodal-recall 0.90 \
  --min-multimodal-answer-correctness 0.80 \
  --min-multimodal-faithfulness 1.0 \
  --min-evidence-hit-rate 0.80 \
  --min-answer-correctness 0.80 \
  --min-faithfulness 1.0 \
  --max-leakage-failures 0 \
  --max-p95-retrieval-ms 800 \
  --max-p95-multimodal-ms 1000 \
  --max-p95-rerank-ms 1500
```

`eval_retrieval.py` 会输出整体 `p95_latency_ms`，并在 `stage_p95_latency_ms` 中记录 `embedding`、`milvus_search`、`rerank`、`context_pack`、`multimodal_search` 等分段 p95；多模态模式下还会细分 `rewrite`、`text_search`、`image_search`、`fusion` 等阶段，便于定位 OCR/caption 通道还是 image vector 通道退化。`release_gate.py` 默认同时检查整体 retrieval p95 和 rerank p95。传入 `--multimodal-input` 后，release gate 会额外运行 `--mode multimodal`，并检查多模态 recall/MRR/nDCG 和 p95 latency；传入 `--multimodal-answer-input` 后，会额外运行 `eval_answer.py --mode multimodal`，检查图片证据回答的 citation、evidence hit、answer correctness 和 faithfulness。

`eval_answer.py --mode multimodal` 会使用 `answer_multimodal.py` 回放图片/OCR/caption 证据回答，继续输出 citation accuracy、evidence hit rate、answer correctness 和 faithfulness。

入库脚本会把 PII 策略处理后的 canonical `SourceDocument` 归档到 `RAG_OBJECT_STORE_DIR/canonical/source_documents.jsonl`，删除 tombstone 归档到 `RAG_OBJECT_STORE_DIR/canonical/deleted_documents.jsonl`。Milvus 只作为检索索引；如果索引损坏或 embedding 模型升级，可以先重建 schema，再执行：

```bash
python projects/08-industrial-rag/rebuild_from_object_store.py --reset
```

本地 `object_store/` 已被 `.gitignore` 忽略。生产中应替换为 S3、MinIO 或企业对象存储，并保存原始文件、解析文本和处理版本。

入库和重建脚本默认会更新 `RAG_OBJECT_STORE_DIR/current_versions.json`。当请求没有显式传 `doc_version` 时，查询链路会读取这个 registry，并把 filter 切到每篇文档的当前发布版本；显式传 `doc_version` 时仍可以检索历史版本。需要只入库不发布时，可以给入库脚本传 `--no-publish-current`。

查询链路也会把当前 `embedding_model` 加入 Milvus filter，避免同一 collection 中残留的旧模型同维度向量被混查。模型升级的推荐方式仍是新建 collection 或新字段；这个 filter 是防止迁移期误召回旧向量的护栏。

Milvus Lite 使用本地数据库文件，不适合多个 Python 进程同时打开同一个 `industrial_rag_demo.db`。本地调试建议串行运行脚本；如果需要显式指定 Lite 文件路径，优先使用 `RAG_MILVUS_URI=/path/to/demo.db`，避免 `pymilvus` 在 import 阶段把文件路径当成 HTTP URI 解析。服务化或多人开发请使用 `docker-compose.yml` 启动 Milvus Standalone，并设置 `MILVUS_URI=http://127.0.0.1:19530`。

本地教学默认让 `IMAGE_EMBEDDING_DIM` 跟 `EMBEDDING_DIM` 一致，保证 Milvus Lite 的多向量索引可跑。生产中如果视觉模型维度不同，可以设置 `IMAGE_EMBEDDING_DIM` 并使用 Milvus Standalone/Cluster 验证多向量索引。

## 1. 系统目标

### 必须具备

- 文本文档 RAG：PDF、Markdown、HTML、纯文本、业务 JSON。
- 图片/图文 RAG：图片 OCR、图片 caption、图片向量检索、图文混合检索。
- Milvus collection 显式 schema，不依赖动态字段承载核心业务字段。
- dense embedding、Milvus BM25、metadata filter、hybrid search、rerank。
- tenant / ACL 权限过滤在检索阶段生效。
- 支持增量写入、文档删除、版本发布和 embedding 模型升级。
- 支持离线评估、线上监控、可回放排障。
- LLM 生成层通过 NewAPI 调用，不把 API key 写入代码或文档。

### 推荐硬件基线

当前机器是 NVIDIA RTX 4060，可作为最小本地开发和轻量上线基线：

- embedding：`BAAI/bge-m3`，FP16，batch size 视显存从 4/8/16 起压测。
- rerank：`BAAI/bge-reranker-v2-m3`，只对召回后的候选做二阶段排序。
- Milvus：开发期可用 Milvus Lite；上线建议 Milvus Standalone 或 Milvus Cluster。
- LLM：通过已有 NewAPI 网关调用，例如 `NEW_API_URL` + `NEW_API_KEY` + `LLM_MODEL`。

不要把真实 API key 提交到仓库。配置示例只保留环境变量名：

```bash
export NEW_API_URL="http://<newapi-host>:<port>"
export NEW_API_KEY="..."
export LLM_MODEL="gemini-3-flash-preview"
```

## 2. 总体架构

```text
                ┌──────────────────────┐
                │      原始知识源       │
                │ PDF/HTML/MD/图片/API │
                └──────────┬───────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────┐
│                  Ingestion Pipeline                 │
│ parse -> clean -> chunk -> enrich metadata -> embed │
└───────────────┬──────────────────────┬─────────────┘
                │                      │
                ▼                      ▼
        ┌──────────────┐       ┌────────────────┐
        │ Object Store │       │    Milvus      │
        │ raw/text/img │       │ vectors+fields │
        └──────────────┘       └───────┬────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────┐
│                   Query Pipeline                    │
│ rewrite -> embed -> hybrid search -> rerank -> pack │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ LLM Answer API  │
              │ citation+guard  │
              └─────────────────┘
```

关键原则：

- 原始文档和解析后的 canonical text 不以 Milvus 为唯一真相源。Milvus 是检索索引，必要时可以重建。
- Milvus 保存检索需要的向量、metadata、短文本和对象引用。
- 权限、租户、版本、语言、来源、时间等 metadata 必须在写入时完整入库。
- 查询阶段先做权限过滤，再召回；不能把无权限结果取回来后再简单丢弃。

## 3. 模型选型

### 文本 Embedding

首选：

```text
BAAI/bge-m3
```

理由：

- 向量维度 1024，最大序列长度 8192。
- 支持多语言，适合中文知识库。
- 支持 dense、sparse lexical weights、ColBERT-style multi-vector 三种检索信号。
- 官方模型卡建议 RAG 使用 hybrid retrieval + reranking。

开发期可先只落地 dense + Milvus BM25，后续再按业务需要接更复杂的 late interaction 模型。

### Reranker

首选：

```text
BAAI/bge-reranker-v2-m3
```

用法定位：

- 不参与全库召回。
- 只对 Milvus 召回后的 top 20-100 候选做 query-document pair 打分。
- 最终取 top 3-8 个 chunk 进入 prompt。

### 多模态 Embedding

推荐分两阶段：

1. 稳定上线版：图片 OCR + image caption 转文本，使用 `bge-m3` 做文本检索。
2. 增强版：引入视觉 embedding，例如 BGE visualized / VISTA 系列或 CLIP 类模型，为图片建立 `image_dense_vector`。

当前仓库状态：

- 已完成并可教学/上线的稳定版：文本 RAG 主链路、多模态 OCR/caption + image vector 融合检索、current-version、eval、monitoring、release gate。
- 已接线但还不是专用 BGE visualized 方案的增强版：`image_dense_vector` 当前支持 CLIP backend，多模态 pipeline 已完整，但尚未落专门的 BGE visualized / VISTA backend。

多模态不要一开始就把所有信号混在一个向量字段里。更稳妥的 schema 是多字段：

```text
text_dense_vector
image_dense_vector
bm25_sparse_vector
```

查询时根据用户输入类型选择不同检索通道，再融合排序。

## 4. Milvus Schema 设计

推荐 collection：

```text
rag_chunks_v1
```

一个 entity 对应一个 chunk，而不是一篇完整文档。

### 字段设计

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | `VARCHAR` primary key | 稳定 chunk 主键，建议由 `tenant_id/doc_id/version/chunk_index` hash 得到 |
| `tenant_id` | `VARCHAR` | 租户隔离 |
| `acl_groups` | `ARRAY<VARCHAR>` | 权限组，查询时使用 `ARRAY_CONTAINS_ANY` 在检索阶段过滤 |
| `doc_id` | `VARCHAR` | 文档 ID |
| `doc_version` | `INT64` | 文档版本 |
| `chunk_index` | `INT64` | chunk 在文档内的位置 |
| `source_type` | `VARCHAR` | `pdf` / `html` / `md` / `image` / `api` |
| `source_uri` | `VARCHAR` | 原始文档或对象存储 URI |
| `title` | `VARCHAR` | 标题或章节标题 |
| `text` | `VARCHAR` | 可直接进入 prompt 的 chunk 文本 |
| `language` | `VARCHAR` | `zh` / `en` / `mixed` |
| `created_at` | `INT64` | 写入时间戳 |
| `updated_at` | `INT64` | 更新时间戳 |
| `is_active` | `BOOL` | 软删除或版本过滤 |
| `embedding_model` | `VARCHAR` | 例如 `BAAI/bge-m3` |
| `embedding_dim` | `INT64` | 例如 1024 |
| `content_hash` | `VARCHAR` | 去重和幂等写入 |
| `text_dense_vector` | `FLOAT_VECTOR(1024)` | BGE-M3 dense embedding |
| `bm25_sparse_vector` | `SPARSE_FLOAT_VECTOR` | Milvus BM25 function 从 `text` 自动生成的关键词权重 |
| `image_dense_vector` | `FLOAT_VECTOR(dim)` | 图片向量，可选 |
| `metadata` | `JSON` | 业务扩展字段 |

### PyMilvus schema 草图

```python
from pymilvus import DataType, MilvusClient

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=False,
)

schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
schema.add_field("doc_version", DataType.INT64)
schema.add_field("chunk_index", DataType.INT64)
schema.add_field("source_type", DataType.VARCHAR, max_length=32)
schema.add_field("source_uri", DataType.VARCHAR, max_length=512)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("text", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
schema.add_field("language", DataType.VARCHAR, max_length=16)
schema.add_field(
    "acl_groups",
    DataType.ARRAY,
    element_type=DataType.VARCHAR,
    max_capacity=32,
    max_length=64,
)
schema.add_field("created_at", DataType.INT64)
schema.add_field("updated_at", DataType.INT64)
schema.add_field("is_active", DataType.BOOL)
schema.add_field("embedding_model", DataType.VARCHAR, max_length=128)
schema.add_field("embedding_dim", DataType.INT64)
schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
schema.add_field("text_dense_vector", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("bm25_sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
schema.add_field("image_dense_vector", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("metadata", DataType.JSON)
```

说明：

- `enable_dynamic_field=False` 是生产默认选择，避免字段拼写错误或脏字段悄悄进入库。
- `text` 开启 analyzer 后可配合 Milvus full text / BM25 能力。
- `image_dense_vector` 的维度由实际视觉 embedding 模型决定；如果还没启用图片向量，可先不建这个字段，或单独建 `rag_image_chunks_v1`。
- 如果 `acl_groups` 查询复杂，建议由权限服务先算出可访问范围，再转成 Milvus filter，避免在向量库里承载过重权限逻辑。

## 5. Index 设计

### 开发期

- dense：`FLAT` 或 `HNSW`。
- sparse：`SPARSE_INVERTED_INDEX`。
- metric：dense 常用 `COSINE` 或向量归一化后的 `IP`；sparse 常用 `IP`。

开发期目标是正确性和可解释性，先用较小数据集验证 recall。

### 上线期

推荐从 HNSW 起步：

```text
text_dense_vector:
  index_type: HNSW
  metric_type: COSINE
  params:
    M: 16-32
    efConstruction: 100-300
  search_params:
    ef: 64-256
```

数据规模明显变大或内存紧张时，再评估 IVF/PQ：

```text
IVF_FLAT:
  nlist: 1024-8192
  nprobe: 16-128

IVF_PQ:
  用于更大规模和更强压缩，但必须重点评估召回损失。
```

调参方法：

1. 用 `FLAT` 建一个小规模准确基线。
2. 固定 eval query 集，记录 `recall@k`、MRR、nDCG、p50/p95/p99 latency。
3. 扫 `ef` 或 `nprobe`，找召回和延迟的折中点。
4. 线上按租户、语种、数据规模分桶观察，不只看总体平均值。

## 6. 写入链路设计

```text
load source
  -> parse
  -> normalize
  -> split chunks
  -> enrich metadata
  -> OCR/caption for images
  -> embedding
  -> batch upsert Milvus
  -> smoke search
  -> publish version
```

### 文档解析

不同来源采用不同 parser：

- PDF：按页提取文本并生成 canonical `SourceDocument`，metadata 保留 `page_no`、`page_start`、`page_end`、`page_count`，prompt 证据头会显示页码用于 citation。
- HTML：去 script/style/nav/header/footer/aside 等非正文内容；保留 URL 和 DOM heading path。
- Markdown：按 heading 切成 section 级 canonical 文档，metadata 保留 `heading_path`，再按 token 长度细切。
- 图片：OCR 提取文字；caption 模型生成语义描述；保留图片 URI。
- 表格：CSV/TSV 转成 compact markdown table，同时保留 `columns`、`row_count_total`、`row_start`、`row_end`、`relative_path` 等结构化 metadata。

### Chunk 策略

建议：

- 中文知识库：以章节为优先边界，再控制 token 数。
- 初始 chunk 大小：400-800 tokens。
- overlap：50-120 tokens。
- 表格和代码块尽量不拆断；当前 `chunk_document` 会先识别 fenced code block、markdown table、段落等结构块，再按 token budget 组合 chunk。
- 单个结构块超过 token budget 时，会按 token 窗口切原文片段，尽量保留代码缩进、表格换行、公式符号和标点；不要把代码先 tokenize 再用空格拼回去。
- 每个 chunk 带上标题路径，例如 `产品手册 > 计费 > 退款规则`。

chunk 文本建议格式：

```text
标题路径: 产品手册 > 计费 > 退款规则
来源: handbook
正文:
...
```

这样 embedding 能看到上下文，reranker 和 LLM 也更容易判断来源。

用固定 eval set 对比 chunk 参数，而不是凭感觉改：

```bash
python projects/08-industrial-rag/sweep_chunking.py \
  --mode hybrid \
  --spec 400:80 \
  --spec 700:100 \
  --spec 1000:150 \
  --json-output projects/08-industrial-rag/runtime/chunk_sweep.jsonl
```

输出会包含每组参数的 `chunk_count`、平均 chunk token 数、recall、MRR、nDCG 和 p95 latency。教学时可以用它解释 chunk 太小导致上下文破碎、chunk 太大导致候选少但噪声更高、overlap 增大导致写入量增加这些 tradeoff。

BGE embedding 路径会在调用模型前检查 tokenizer 后的输入长度；如果 chunk 超过 `RAG_EMBED_MAX_LENGTH`，脚本会失败并要求重新分块，而不是静默截断后半段文本。这样可以避免“入库成功但重要证据被截断丢失”的隐性数据质量问题。

每组参数会使用独立的临时 Milvus collection；评估结束后脚本会 drop 临时 collection，并在输出行中记录 `temporary_collection` 和 `cleanup`，避免 chunk sweep 长期污染本地或测试 Milvus。

### 幂等写入

稳定主键：

```text
chunk_id = sha256(tenant_id + doc_id + doc_version + chunk_index)
```

内容去重：

```text
content_hash = sha256(normalized_text)
```

更新策略：

- 同版本重复写入：upsert。
- 新版本发布：写入 `doc_version=N+1`，更新 `current_versions.json`，查询 filter 默认切到新版本。
- 删除文档：按 `doc_id` 批量 delete，或 `is_active=false` 软删除。
- embedding 模型升级：新建 collection 或新字段，不要混用不同向量空间；查询 filter 会限制当前 `embedding_model`，避免迁移期混查旧模型向量。

## 7. 查询链路设计

```text
user query
  -> auth context
  -> query normalize
  -> optional query rewrite
  -> dense embedding
  -> BM25 query text
  -> Milvus hybrid search with metadata filter
  -> bge reranker
  -> context packing
  -> LLM answer
  -> citation and trace log
```

### Query Rewrite

只在必要时做：

- 用户问题过短。
- 多轮对话需要补全指代。
- 需要把口语化问题改写成业务检索词。

rewrite 不能引入权限外信息，必须保留原 query 和 rewritten query 方便回放。

### Metadata Filter

典型 filter：

```text
tenant_id == "team_a"
and is_active == true
and doc_version in [current_version]
and source_type in ["pdf", "md", "html"]
```

权限字段必须来自服务端认证上下文，不能相信用户 query 里的租户或权限声明。

### Hybrid Search

推荐召回通道：

1. `text_dense_vector`：语义召回。
2. `bm25_sparse_vector` 或 Milvus full text：关键词、代码、错误码、专有名词召回。
3. `image_dense_vector`：图片相似检索或图文检索。

`bm25_sparse_vector` 由 Milvus BM25 function 从 `text` 字段自动生成；写入 entity 时不再手工传 sparse dict。查询 BM25/hybrid 时直接传 query 文本，Milvus 按同一 analyzer 生成查询端 BM25 表示。

Milvus 多向量 hybrid search 的思路：

```python
dense_req = AnnSearchRequest(
    data=[dense_query_vector],
    anns_field="text_dense_vector",
    param={"metric_type": "COSINE", "params": {"ef": 128}},
    limit=50,
    expr=filter_expr,
)

sparse_req = AnnSearchRequest(
    data=[query_text],
    anns_field="bm25_sparse_vector",
    param={"metric_type": "BM25"},
    limit=50,
    expr=filter_expr,
)

results = client.hybrid_search(
    collection_name="rag_chunks_v1",
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(),
    limit=50,
    output_fields=["text", "doc_id", "title", "source_uri", "chunk_index"],
)
```

融合策略：

- `RRFRanker`：默认优先，适合 dense/sparse 分数尺度不同。
- `WeightedRanker`：当你已经通过评估知道各通道权重时使用。

### BGE Rerank

rerank 输入：

```text
[(query, chunk_text_1), (query, chunk_text_2), ...]
```

排查 rerank 前后的排序变化时，先运行：

```bash
python projects/08-industrial-rag/diagnose_retrieval.py \
  "RAG 检索变慢时应该排查什么" \
  --tenant-id team_a \
  --acl-group ops \
  --json-output projects/08-industrial-rag/runtime/retrieval_diagnosis.jsonl
```

输出会列出每个候选在 dense、sparse、hybrid、rerank 中的 rank/score，以及 query 与 chunk 的 lexical overlap。教学时可以用它解释：正确文档是没有被召回，还是召回了但被 reranker 降下去了。

候选数量：

- Milvus hybrid 召回：20-100。
- reranker 后保留：3-8。

规则：

- reranker 只处理权限过滤后的候选。
- rerank 分数低于阈值时，可以拒答或改走澄清问题。
- 最终 prompt 不要只按分数塞满，要去重、控制同文档 chunk 数量、保留 citation。

排查最终 prompt 为什么没有某条证据时，运行：

```bash
python projects/08-industrial-rag/diagnose_context.py \
  "RAG 检索变慢时应该排查什么" \
  --tenant-id team_a \
  --acl-group ops \
  --max-context-chars 1200 \
  --max-chunks-per-doc 1 \
  --json-output projects/08-industrial-rag/runtime/context_diagnosis.jsonl
```

输出会给出每个 reranked candidate 的 `select/drop`、原因和当时已使用的 context 字符数。常见 drop 原因包括 `below_min_rerank_score`、`max_chunks_per_doc`、`context_char_budget`、`context_hit_limit`。如果高分候选因为同文档 chunk 限制或分数阈值被丢掉，context packer 会继续从后续候选补位，而不是先截成固定 topK 再放弃。

## 8. 多模态 RAG 设计

### 图片入库

图片不要只存向量，至少保留：

```text
image_uri
ocr_text
caption
page_no
bbox
linked_doc_id
image_dense_vector
```

推荐流程：

```text
image
  -> OCR text
  -> visual caption
  -> image embedding
  -> text embedding for OCR/caption
  -> write Milvus
```

### 图片查询

文本查图片：

- query 用文本 embedding 检索 `caption` / `ocr_text` 的 dense vector。
- 同时检索 `image_dense_vector`，如果视觉模型支持 text-image shared embedding。
- 当前 `search_multimodal.py` 会同时跑 OCR/caption 的 text hybrid 通道和 `image_dense_vector` 通道，再用 RRF 融合；输出 metadata 中的 `fusion.channels` 会标明每条证据来自哪些通道。

图片查文档：

- 上传图片生成 image embedding。
- 检索 `image_dense_vector`。
- 再用图片 caption/OCR 文本补充 text hybrid search。

### Prompt 组织

LLM 如果支持多模态输入：

- 传入图片 URI 或 base64。
- 同时传入 OCR/caption 和 citation。

LLM 如果只支持文本：

- 使用 OCR/caption 作为文本证据。
- 明确提示“图片证据来自 OCR/caption，可能不完整”。

当前 `build_prompt()` 会在图片证据头中保留 `source_type=image`、`image_uri`、`linked_doc_id`、页码/行号/bbox 等定位信息，并在 prompt 规则中加入图片证据限制提示，避免模型把 OCR/caption 当成完整原图事实。

需要直接基于图片证据回答时，使用 `answer_multimodal.py`。它会先运行 `search_multimodal.py` 的 OCR/caption text hybrid + image vector 融合检索，再做 context packing，并复用 `answer.py` 的 NewAPI 生成层；没有配置 `NEW_API_URL` / `NEW_API_KEY` 时会直接失败。

## 9. LLM Answer API

NewAPI 兼容 OpenAI SDK，因此代码里使用 `openai.OpenAI` 客户端：

```python
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["NEW_API_URL"].rstrip("/") + "/v1",
    api_key=os.environ["NEW_API_KEY"],
)

response = client.chat.completions.create(
    model=os.environ.get("LLM_MODEL", "gemini-3-flash-preview"),
    messages=[
        {"role": "system", "content": "你是企业知识库问答助手。只根据给定证据回答。"},
        {"role": "user", "content": prompt},
    ],
)
```

Prompt 必须包含：

- 用户问题。
- 检索证据列表。
- 每条证据的 `doc_id`、`title`、`source_uri`、`chunk_index`。
- 回答规则：证据不足就说不知道，不要编造。

示例结构：

```text
问题:
{query}

证据:
[1] doc_id=..., title=..., chunk=...
...

要求:
- 只使用证据回答。
- 每个关键结论后标注引用编号。
- 如果证据不足，回答“当前知识库没有足够证据”。
```

## 10. 评估体系

### 检索评估

需要构造 eval set：

```text
query
tenant_id
expected_doc_ids
expected_chunk_ids
answerable
query_type
acl_groups
doc_version
source_types
history
expected_answer_terms
unsupported_answer_terms
```

`expected_doc_ids` 用于文档级 recall/MRR/nDCG；如果提供 `expected_chunk_ids`，则切换为 chunk 级评估。chunk id 支持 Milvus 主键、metadata 中的 `chunk_id`，或教学更直观的 `doc_id:chunk_index` 格式。`acl_groups`、`doc_version`、`source_types`、`history` 会在 `eval_retrieval.py` 和 `eval_answer.py` 中按原请求条件回放。

`expected_answer_terms` 用于教学版 answer correctness：答案必须覆盖这些关键术语。`unsupported_answer_terms` 用于教学版 faithfulness：如果答案出现这些术语但证据没有出现，会被视为未被证据支持。生产中可以把这两个规则指标替换或补充为人工标注、LLM judge、NLI/entailment 检查。

`build_eval_from_feedback.py` 会从 runtime 事件导出同样格式的 JSONL，并额外保留 `source_request_id`、`feedback_rating`、`feedback_comment` 等排障字段；eval 脚本会忽略这些附加字段。

指标：

- `recall@k`：正确 chunk 是否进入候选。
- `MRR`：第一个正确结果排名。
- `nDCG@k`：多相关等级排序质量。
- hybrid / multimodal ablation：`eval_retrieval.py --mode dense|sparse|hybrid|rerank|multimodal`，对比 dense only、sparse only、hybrid、hybrid+rerank，以及 OCR/caption text hybrid + image vector 融合检索。
- latency：embedding、Milvus search、rerank、LLM 分段统计。

### 回答评估

指标：

- answer correctness：答案是否正确。
- faithfulness：是否被证据支持。
- citation accuracy：引用是否指向正确 chunk。
- refusal quality：证据不足时是否拒答。
- leakage test：是否返回无权限租户内容。

上线门槛示例：

```text
recall@50 >= 0.90
rerank recall@8 >= 0.80
权限泄露测试 = 0
p95 retrieval latency <= 800ms
p95 rerank latency <= 1500ms
```

具体阈值必须按业务和硬件压测确定，不能照抄。

## 11. 部署方案

### 服务拆分

```text
rag-api
  /query
  /feedback
  /health

ingestion-worker
  parse/chunk/embed/upsert

embedding-service
  BAAI/bge-m3

rerank-service
  BAAI/bge-reranker-v2-m3

milvus
  standalone or cluster

object-store
  raw docs/images
```

### 4060 本地部署建议

- embedding 和 rerank 可以共用一张卡，但不要同时无限并发。
- 用队列控制 ingestion embedding batch。
- 查询链路给 rerank 设置最大候选数和超时。
- 使用 FP16；必要时评估 ONNX / TensorRT / quantization。
- 如果模型下载 Hugging Face 403，可使用国内镜像，但镜像地址和 token 不要硬编码到代码。

### 配置项

```text
RAG_MILVUS_URI
MILVUS_URI
MILVUS_TOKEN
RAG_COLLECTION
RAG_OBJECT_STORE_DIR
RAG_REQUIRE_AUTH_CONTEXT
RAG_API_TOKEN
EMBEDDING_MODEL=BAAI/bge-m3
RERANK_MODEL=BAAI/bge-reranker-v2-m3
NEW_API_URL
NEW_API_KEY
LLM_MODEL
HF_ENDPOINT
```

## 12. 监控与排障

### Trace 必须记录

- `request_id`
- 原始 query / rewritten query
- auth context 摘要
- filter expr
- embedding model version
- Milvus raw hits
- rerank hits
- final context
- LLM model / token usage / latency
- answer citation

不要记录完整 API key、用户隐私字段、未脱敏个人信息。

### 常见问题

| 现象 | 可能原因 | 排查 |
| --- | --- | --- |
| 召回不到正确文档 | chunk 太大/太小、embedding 模型不适配、filter 过严 | 跑 `sweep_chunking.py` 对比 chunk 参数，再用 `diagnose_retrieval.py` 看 dense/sparse/hybrid rank |
| dense 命中差但关键词明显 | 专有名词/错误码问题 | 加 BM25/sparse，调 hybrid 权重 |
| rerank 后变差 | 候选质量差、reranker 不适配领域、文本截断 | 用 `diagnose_retrieval.py` 对比 hybrid_rank 和 rerank_rank，检查 rerank 输入文本 |
| 回答编造 | prompt 证据不足、没有拒答规则 | 跑 `diagnose_context.py` 看证据是否被 budget/阈值丢弃，加 faithfulness 评估和证据阈值 |
| 延迟高 | topK 太大、rerank 候选太多、index 参数过高 | 分段看 p95，调 `ef/nprobe/topK` |
| 权限泄露 | filter 不完整或后置过滤 | 检查 auth context 到 Milvus filter 的映射 |

## 13. 安全与权限

必须做到：

- 服务端生成 metadata filter。
- 不信任用户输入的 `tenant_id`、`doc_id`、`acl`；生产 API 使用请求头或网关注入的 auth context。
- API key 只通过环境变量或 secret manager 注入。
- 日志脱敏。
- 每次回答保留 citation，便于审计。
- 对跨租户 query 做自动化测试。

权限策略：

```text
user -> auth service -> allowed_tenant_ids + allowed_acl_groups
     -> build Milvus filter
     -> search only accessible chunks
```

## 14. 后续代码落地顺序

建议按以下顺序实现，不要一开始就堆满所有能力：

1. `schema.py`：显式创建 Milvus collection 和 index。
2. `ingest_text.py`：Markdown/PDF 文本解析、chunk、BGE-M3 dense embedding、写入 Milvus。
3. `search_dense.py`：metadata filter + dense search。
4. `search_hybrid.py`：BM25/sparse + dense hybrid search。
5. `rerank.py`：接入 `bge-reranker-v2-m3`。
6. `answer.py`：通过 NewAPI 生成带 citation 的回答。
7. `ingest_image.py`：OCR/caption/image embedding。
8. `eval_retrieval.py`：构造 eval set，输出 recall/MRR/nDCG/latency。
9. `serve.py`：FastAPI 查询服务。
10. `docker-compose.yml`：Milvus + RAG API + worker 的最小部署。

## 15. 参考资料

- Milvus Multi-Vector Hybrid Search: https://milvus.io/docs/multi-vector-search.md
- Milvus Full Text Search / BM25: https://milvus.io/docs/full-text-search.md
- Milvus Reranking: https://milvus.io/docs/reranking.md
- Milvus Schema: https://milvus.io/docs/v2.6.x/schema.md
- Milvus Index Explained: https://milvus.io/docs/index-explained.md
- BAAI/bge-m3 model card: https://huggingface.co/BAAI/bge-m3
- BGE-M3 documentation: https://bge-model.com/bge/bge_m3.html
- BAAI/bge-reranker-v2-m3 model card: https://huggingface.co/BAAI/bge-reranker-v2-m3
- BAAI/bge-visualized: https://huggingface.co/BAAI/bge-visualized
