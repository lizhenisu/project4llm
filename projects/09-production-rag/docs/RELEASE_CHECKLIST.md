# Industrial RAG Release Checklist

目标：用这份清单验证 09 是否达到可上线、可观测、可回归的生产发布标准。

## 1. 本地生产回归验收

```bash
source ../../.venv/bin/activate
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
make release-gate-smoke
make deploy-contract-smoke
make pii-scan
make eval
make answer-eval
make release-gate
make benchmark
make monitor
make export-feedback-eval
make deploy-smoke
make check
make milvus-smoke
```

通过标准：

- `smoke_e2e=ok`
- `smoke_api=ok`
- `smoke_api_multimodal=ok`
- `smoke_readiness=ok`
- `/ready` 的 schema 检查会验证必需字段、文本/图片向量维度，以及 `text` 字段启用 analyzer
- `smoke_event_redaction=ok`
- `recall@5 >= 1.000`，样例集权限泄露为 0
- `ndcg@5`、`mrr@5` 输出正常
- `citation_accuracy`、`evidence_hit_rate`、`refusal_quality` 输出正常
- `smoke_benchmark=ok`
- `benchmark_latency.py` 会回放真实 text / multimodal answer pipeline，并输出 rewrite/search/rerank/context/answer 分段延迟
- `runtime/*.jsonl` 生成 retrieval、answer、feedback 事件
- `smoke_security=ok`
- `smoke_auth_context=ok`
- `smoke_context=ok`
- `smoke_context_diagnosis=ok`
- `smoke_context_backfill=ok`
- `smoke_rewrite=ok`
- `smoke_answer_eval=ok`
- `smoke_answer_quality_eval=ok`
- `smoke_file_ingest=ok`
- `smoke_table_ingest=ok`
- `smoke_chunk_structure=ok`
- `smoke_chunk_sweep=ok`
- `smoke_search_params=ok`
- `smoke_multimodal_search=ok`
- `smoke_multimodal_prompt=ok`
- `smoke_multimodal_eval=ok`
- `smoke_multimodal_answer=ok`
- `smoke_multimodal_answer_eval=ok`
- `smoke_pdf_page_metadata=ok`
- `smoke_heading_metadata=ok`
- `smoke_feedback_eval_export=ok`
- `smoke_eval_filters=ok`
- `smoke_sparse_ablation=ok`
- `smoke_retrieval_diagnosis=ok`
- `smoke_lifecycle=ok`
- `smoke_current_version=ok`
- `smoke_current_version_unpublish=ok`
- `smoke_embedding_model_filter=ok`
- `smoke_source_filter=ok`
- `smoke_object_store_rebuild=ok`
- `smoke_object_store_delete_tombstone=ok`
- `smoke_observability=ok`
- `smoke_monitoring=ok`
- `smoke_release_gate=ok`
- 跨租户 query 的 trace 为 `blocked_cross_tenant_query`
- `scan_pii.py --fail` 在样例数据上通过
- PDF/HTML/Markdown/TXT 目录可通过 `ingest_files.py` 统一入库
- CSV/TSV 表格可通过 `ingest_tables.py` 转 compact markdown table 入库，并保留列名、行数、行范围和来源路径 metadata
- chunk 过程会保留 markdown table 和 fenced code block 的结构，不会先把全文压成一行再切碎
- `sweep_chunking.py` 能用隔离 collection 对多组 chunk_size/overlap 输出 chunk 数、recall/MRR/nDCG 和 p95 latency
- context packing 会在 `max_chunks_per_doc`、分数阈值或命中上限生效时继续从后续候选补位，而不是先把候选截断成固定 topK
- HNSW `M/efConstruction/ef` 和 sparse `drop_ratio_*` 参数能通过环境变量配置；建索引参数和查询参数的生效边界在 README 中明确说明
- 多模态检索会融合 OCR/caption text hybrid 通道和 `image_dense_vector` 通道，并在结果 metadata 中记录通道 rank
- 多模态 `/search` 和 `/query` 会把图片证据 metadata 原样带回服务响应，包括 `image_uri`、`bbox`、`linked_doc_id`、`fusion.channels`
- 图片证据进入 prompt 时会保留 `source_type=image`、`image_uri`、`linked_doc_id`、bbox 等定位字段，并提示 OCR/caption 可能不完整
- `answer_multimodal.py` 能把多模态融合检索结果送入 context packing 和 answer generation，并返回带 citation 的图片证据答案
- `eval_answer.py --mode multimodal` 能评估图片证据回答的 citation、evidence hit、answer correctness 和 faithfulness
- `/search` 和 `/query` 支持 `query_mode=multimodal`，服务化入口可直接走图文融合检索和回答链路
- 多模态 `/search`、`/query`、`eval_retrieval.py --mode multimodal`、`eval_answer.py --mode multimodal` 都会按原请求回放 `history`
- PDF 入库会按页保留 `page_no/page_start/page_end/page_count`，prompt 证据头能显示页码，便于 citation 审计
- Markdown/HTML 入库会保留 heading path；Markdown 按 heading section 入 canonical 文档，HTML 会过滤常见非正文标签
- `doc_version` 过滤能命中指定版本，并且不存在版本不返回证据
- 无显式 `doc_version` 时默认只检索 current-version registry 中的发布版本；显式 `doc_version` 可查历史版本
- 删除当前发布版本时会同步取消 current-version registry；删除非当前历史版本不会误改发布状态
- 查询 filter 包含当前 `embedding_model`，迁移期间不会召回旧模型写入的同维度向量
- `source_types` metadata filter 能在 API 和 pipeline 中限制 `md/html/pdf/image/api` 等来源类型
- 入库后的 canonical text 归档到 object store，`rebuild_from_object_store.py --reset` 可重建 Milvus 索引
- 删除 tombstone 归档到 object store，重建索引时不会复活已删除文档；重新入库同一文档版本会清理对应 tombstone
- runtime 事件包含 `raw_hits`、`rerank_hits`、`final_context`、`trace.stage_latency_ms` 和 `llm.latency_ms`
- `monitor_events.py` 能输出 runtime 事件的 p50/p95/p99 latency、retrieval mode、source type、fusion channel、context、top docs 和 feedback rating 分布
- `build_eval_from_feedback.py` 能把 feedback 与 retrieval/answer 事件合并成 `eval_retrieval.py` / `eval_answer.py` 可读的 JSONL
- `eval_retrieval.py` / `eval_answer.py` 能按 eval 样本中的 `acl_groups`、`doc_version`、`source_types`、`history` 回放，并支持 `expected_chunk_ids`
- `eval_retrieval.py --mode dense|sparse|hybrid|rerank|multimodal` 能做完整 hybrid 和多模态 ablation
- `diagnose_retrieval.py` 能输出 dense/sparse/hybrid/rerank rank 和 score，用于解释召回失败或 rerank 退化
- `diagnose_context.py` 能输出每个 reranked candidate 进入或离开 final context 的原因
- `release_gate.py` 能实际调用 `eval_retrieval.py` / `eval_answer.py`，输出 JSON report，并按 recall、MRR、nDCG、权限泄露、p95 retrieval、p95 rerank、citation、evidence hit、refusal quality、answer correctness、faithfulness 阈值失败退出；传 `--multimodal-input` 后还能额外检查多模态 recall/MRR/nDCG 和 p95 latency；传 `--multimodal-answer-input` 后还能额外检查图片证据回答质量

## 2. Milvus Standalone 验收

```bash
docker compose up -d
export MILVUS_URI=http://127.0.0.1:19530
make schema
make ingest
make smoke
make milvus-smoke
make benchmark
```

容器化 API 验收：

```bash
docker compose up -d milvus rag-api rag-web
docker compose --profile ingest run --rm rag-ingest
RAG_API_URL=http://127.0.0.1:8080/api make deploy-smoke
```

如果生产 API 启用 header auth context：

```bash
export RAG_REQUIRE_AUTH_CONTEXT=1
export RAG_API_TOKEN=dev-only-token
export RAG_DEPLOY_TENANT_ID=team_a
export RAG_DEPLOY_ACL_GROUPS=ops,support
RAG_API_URL=http://127.0.0.1:8080/api make deploy-smoke
```

通过标准：

- `smoke_milvus=ok`
- `rag_chunks_v1` collection 存在
- `/ready` 返回 `status=ok`，schema 字段和向量维度与当前配置一致
- hybrid search、ACL filter、image search 均能返回预期结果
- `smoke_deploy_contract=ok`
- `smoke_deploy=ok`；设置 `RAG_API_URL` 时验证外部部署，未设置时启动隔离 API 进程覆盖 `/ready`、`/search`、`/query`、`/feedback` 和可选 auth header

## 3. 托管模型验收

```bash
export RAG_EMBEDDING_BACKEND=siliconflow
export RAG_RERANK_BACKEND=siliconflow
export EMBEDDING_MODEL=BAAI/bge-m3
export RERANK_MODEL=BAAI/bge-reranker-v2-m3
export SILICONFLOW_API_KEY=...
export RAG_RUN_MODEL_SMOKE=1
make model-smoke
```

图片向量使用 CLIP 验收：

```bash
export RAG_IMAGE_EMBEDDING_BACKEND=none
export IMAGE_EMBEDDING_MODEL=disabled-image-embedding
export IMAGE_EMBEDDING_DIM=512
export RAG_RUN_MODEL_SMOKE=1
make model-smoke
```

通过标准：

- embedding 向量维度和 schema 一致
- reranker 能输出分数
- 默认纯 API 部署不会加载本地 CLIP/torch 权重；只有显式设置 `RAG_IMAGE_EMBEDDING_BACKEND=clip` 时才验证 CLIP 维度

## 4. LLM 网关验收

```bash
export NEW_API_URL=http://<newapi-host>:<port>
export NEW_API_KEY=...
export LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
make llm-smoke
python answer.py "RAG 检索变慢时应该排查什么" --tenant-id team_a --acl-group ops
```

通过标准：

- `smoke_llm=ok`
- 回答包含 citation 编号
- 未检索到足够证据时能够拒答

## 5. 安全检查

```bash
rg -n "NEW_API_KEY=.*[A-Za-z0-9_-]{20,}|172\\.31\\." .
git status --ignored --short runtime production_rag.db volumes
```

通过标准：

- 真实 API key、内网地址不在仓库文件中
- runtime、Milvus Lite DB、Docker volumes 被 git 忽略
- API 返回 citations，但不返回密钥或用户隐私字段
- runtime 事件会脱敏 query、feedback comment、metadata 和 `text_preview` 中的 PII/API key 形态
- 生产入库使用 `RAG_PII_POLICY=redact` 或 `RAG_PII_POLICY=fail`
- 生产 API 设置 `RAG_REQUIRE_AUTH_CONTEXT=1`，并由网关或认证服务注入 `X-RAG-Tenant-ID` / `X-RAG-ACL-Groups`

## 6. 上线前仍需确认

- 根据真实业务 eval set 重新计算 recall、MRR、nDCG。
- 根据 4060 显存压测 embedding/rerank batch size。
- 根据 Milvus Standalone/Cluster 延迟曲线调整 HNSW `ef`、`M`。
- 将 feedback/retrieval/answer events 从本地 JSONL 切换到数据库或消息队列。
- 接入真实认证服务，由服务端生成 `tenant_id` 和 `acl_groups`。
