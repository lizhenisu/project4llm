# 08-industrial-rag

目标：设计一个带教学目的、但按可上线标准组织的工业级 RAG 向量检索系统。向量库使用 Milvus，文本 embedding 和 rerank 使用 BGE 家族，并预留图片等多模态 RAG 能力。

本项目不是“最小 demo”。它要回答的是：如果要把 RAG 做到生产环境，schema、metadata、hybrid search、rerank、权限、评估、部署和运维应该怎么设计。

## 1. 系统目标

### 必须具备

- 文本文档 RAG：PDF、Markdown、HTML、纯文本、业务 JSON。
- 图片/图文 RAG：图片 OCR、图片 caption、图片向量检索、图文混合检索。
- Milvus collection 显式 schema，不依赖动态字段承载核心业务字段。
- dense embedding、sparse/BM25、metadata filter、hybrid search、rerank。
- tenant / ACL 权限过滤在检索阶段生效。
- 支持增量写入、文档删除、版本发布和 embedding 模型升级。
- 支持离线评估、线上监控、可回放排障。
- LLM 生成层通过 OpenAI-compatible API 调用，不把 API key 写入代码或文档。

### 推荐硬件基线

当前机器是 NVIDIA RTX 4060，可作为最小本地开发和轻量上线基线：

- embedding：`BAAI/bge-m3`，FP16，batch size 视显存从 4/8/16 起压测。
- rerank：`BAAI/bge-reranker-v2-m3`，只对召回后的候选做二阶段排序。
- Milvus：开发期可用 Milvus Lite；上线建议 Milvus Standalone 或 Milvus Cluster。
- LLM：通过已有 NewAPI 网关调用，例如 `OPENAI_BASE_URL` + `OPENAI_API_KEY` + `LLM_MODEL`。

不要把真实 API key 提交到仓库。配置示例只保留环境变量名：

```bash
export OPENAI_BASE_URL="http://<newapi-host>:<port>/v1"
export OPENAI_API_KEY="..."
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

开发期可先只落地 dense + BM25，后续再接 BGE-M3 sparse 和 ColBERT late interaction。

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
| `acl_groups` | `ARRAY<VARCHAR>` 或 `JSON` | 权限组，生产中也可拆到权限服务 |
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
| `bm25_sparse_vector` | `SPARSE_FLOAT_VECTOR` | BM25 或 BGE-M3 sparse lexical weights |
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
schema.add_field("created_at", DataType.INT64)
schema.add_field("updated_at", DataType.INT64)
schema.add_field("is_active", DataType.BOOL)
schema.add_field("embedding_model", DataType.VARCHAR, max_length=128)
schema.add_field("embedding_dim", DataType.INT64)
schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
schema.add_field("text_dense_vector", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("bm25_sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
schema.add_field("image_dense_vector", DataType.FLOAT_VECTOR, dim=768)
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

- PDF：提取文本、标题层级、页码；保留页码用于 citation。
- HTML：去导航、广告、脚注；保留 URL 和 DOM heading。
- Markdown：按 heading 切结构，再按 token 长度细切。
- 图片：OCR 提取文字；caption 模型生成语义描述；保留图片 URI。
- 表格：转成 compact markdown table，同时保留结构化 metadata。

### Chunk 策略

建议：

- 中文知识库：以章节为优先边界，再控制 token 数。
- 初始 chunk 大小：400-800 tokens。
- overlap：50-120 tokens。
- 表格和代码块尽量不拆断。
- 每个 chunk 带上标题路径，例如 `产品手册 > 计费 > 退款规则`。

chunk 文本建议格式：

```text
标题路径: 产品手册 > 计费 > 退款规则
来源: handbook
正文:
...
```

这样 embedding 能看到上下文，reranker 和 LLM 也更容易判断来源。

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
- 新版本发布：写入 `doc_version=N+1`，查询 filter 切到新版本。
- 删除文档：按 `doc_id` 批量 delete，或 `is_active=false` 软删除。
- embedding 模型升级：新建 collection 或新字段，不要混用不同向量空间。

## 7. 查询链路设计

```text
user query
  -> auth context
  -> query normalize
  -> optional query rewrite
  -> dense embedding
  -> sparse/BM25 query
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
    data=[sparse_query_vector],
    anns_field="bm25_sparse_vector",
    param={"metric_type": "IP"},
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

候选数量：

- Milvus hybrid 召回：20-100。
- reranker 后保留：3-8。

规则：

- reranker 只处理权限过滤后的候选。
- rerank 分数低于阈值时，可以拒答或改走澄清问题。
- 最终 prompt 不要只按分数塞满，要去重、控制同文档 chunk 数量、保留 citation。

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

## 9. LLM Answer API

使用 OpenAI-compatible 客户端：

```python
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
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
```

指标：

- `recall@k`：正确 chunk 是否进入候选。
- `MRR`：第一个正确结果排名。
- `nDCG@k`：多相关等级排序质量。
- hybrid ablation：dense only / sparse only / hybrid / hybrid+rerank。
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
MILVUS_URI
MILVUS_TOKEN
RAG_COLLECTION
EMBEDDING_MODEL=BAAI/bge-m3
RERANK_MODEL=BAAI/bge-reranker-v2-m3
OPENAI_BASE_URL
OPENAI_API_KEY
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
| 召回不到正确文档 | chunk 太大/太小、embedding 模型不适配、filter 过严 | 看 raw hits、放宽 filter、做 FLAT baseline |
| dense 命中差但关键词明显 | 专有名词/错误码问题 | 加 BM25/sparse，调 hybrid 权重 |
| rerank 后变差 | 候选质量差、reranker 不适配领域、文本截断 | 看 rerank 输入，做 ablation |
| 回答编造 | prompt 证据不足、没有拒答规则 | 加 faithfulness 评估和证据阈值 |
| 延迟高 | topK 太大、rerank 候选太多、index 参数过高 | 分段看 p95，调 `ef/nprobe/topK` |
| 权限泄露 | filter 不完整或后置过滤 | 检查 auth context 到 Milvus filter 的映射 |

## 13. 安全与权限

必须做到：

- 服务端生成 metadata filter。
- 不信任用户输入的 `tenant_id`、`doc_id`、`acl`。
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
6. `answer.py`：通过 NewAPI/OpenAI-compatible API 生成带 citation 的回答。
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
