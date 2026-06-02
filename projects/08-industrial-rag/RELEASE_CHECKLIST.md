# Industrial RAG Release Checklist

目标：把教学可跑版本推进到可上线版本前，用这份清单逐项验收。

## 1. 本地教学验收

```bash
source ../../.venv/bin/activate
make schema
make ingest
make smoke
make api-smoke
make security-smoke
make context-smoke
make rewrite-smoke
make answer-eval-smoke
make pii-scan
make eval
make answer-eval
make benchmark
make check
make milvus-smoke
```

通过标准：

- `smoke_e2e=ok`
- `smoke_api=ok`
- `recall@5 >= 1.000`，样例集权限泄露为 0
- `ndcg@5`、`mrr@5` 输出正常
- `citation_accuracy`、`evidence_hit_rate`、`refusal_quality` 输出正常
- `benchmark_latency.py` 输出 embedding/search/rerank/answer 分段延迟
- `runtime/*.jsonl` 生成 retrieval、answer、feedback 事件
- `smoke_security=ok`
- `smoke_context=ok`
- `smoke_rewrite=ok`
- `smoke_answer_eval=ok`
- 跨租户 query 的 trace 为 `blocked_cross_tenant_query`
- `scan_pii.py --fail` 在样例数据上通过

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
docker compose up -d milvus rag-api
docker compose --profile ingest run --rm rag-ingest
RAG_API_URL=http://127.0.0.1:8008 make deploy-smoke
```

通过标准：

- `smoke_milvus=ok`
- `rag_chunks_v1` collection 存在
- hybrid search、ACL filter、image search 均能返回预期结果
- `smoke_deploy=ok`

## 3. 真实模型验收

```bash
export HF_ENDPOINT=https://hf-mirror.com
export RAG_EMBEDDING_BACKEND=bge
export RAG_RERANK_BACKEND=bge
export EMBEDDING_MODEL=BAAI/bge-m3
export RERANK_MODEL=BAAI/bge-reranker-v2-m3
export RAG_RUN_MODEL_SMOKE=1
make model-smoke
```

图片向量使用 CLIP 验收：

```bash
export RAG_IMAGE_EMBEDDING_BACKEND=clip
export IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32
export IMAGE_EMBEDDING_DIM=512
export RAG_RUN_MODEL_SMOKE=1
make model-smoke
```

通过标准：

- embedding 向量维度和 schema 一致
- reranker 能输出分数
- CLIP text/image embedding 维度和 `IMAGE_EMBEDDING_DIM` 一致

## 4. LLM 网关验收

```bash
export OPENAI_BASE_URL=http://<newapi-host>:<port>/v1
export OPENAI_API_KEY=...
export LLM_MODEL=gemini-3-flash-preview
make llm-smoke
python answer.py "RAG 检索变慢时应该排查什么" --tenant-id team_a --acl-group ops
```

通过标准：

- `smoke_llm=ok`
- 回答包含 citation 编号
- 未检索到足够证据时能够拒答

## 5. 安全检查

```bash
rg -n "OPENAI_API_KEY=.*[A-Za-z0-9_-]{20,}|172\\.31\\." .
git status --ignored --short runtime industrial_rag_demo.db volumes
```

通过标准：

- 真实 API key、内网地址不在仓库文件中
- runtime、Milvus Lite DB、Docker volumes 被 git 忽略
- API 返回 citations，但不返回密钥或用户隐私字段
- 生产入库使用 `RAG_PII_POLICY=redact` 或 `RAG_PII_POLICY=fail`

## 6. 上线前仍需确认

- 根据真实业务 eval set 重新计算 recall、MRR、nDCG。
- 根据 4060 显存压测 embedding/rerank batch size。
- 根据 Milvus Standalone/Cluster 延迟曲线调整 HNSW `ef`、`M`。
- 将 feedback/retrieval/answer events 从本地 JSONL 切换到数据库或消息队列。
- 接入真实认证服务，由服务端生成 `tenant_id` 和 `acl_groups`。
