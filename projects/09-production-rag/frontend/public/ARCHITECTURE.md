# 09 Production RAG — System Architecture

> **基于 Milvus 的企业级多模态 RAG 知识库系统**

---

## 目录

1. [概览](#1-概览)
2. [系统架构图](#2-系统架构图)
3. [部署架构](#3-部署架构)
4. [数据流全景](#4-数据流全景)
5. [文档摄取管道](#5-文档摄取管道)
6. [多模态处理](#6-多模态处理)
7. [检索管道](#7-检索管道)
8. [重排序](#8-重排序)
9. [上下文打包与答案生成](#9-上下文打包与答案生成)
10. [Milvus 模式设计](#10-milvus-模式设计)
11. [LLM 调用全景图](#11-llm-调用全景图)
12. [对象存储与版本管理](#12-对象存储与版本管理)
13. [认证与授权](#13-认证与授权)
14. [对话管理](#14-对话管理)
15. [Studio：思维导图与数据表格](#15-studio思维导图与数据表格)
16. [评估框架与发布门禁](#16-评估框架与发布门禁)
17. [配置参考](#17-配置参考)
18. [开发与生产环境](#18-开发与生产环境)

---

## 1. 概览

本项目是一个**生产级 RAG（检索增强生成）知识库系统**，支持：

- **多模态检索**：文本 + 图片联合召回
- **混合检索**：Dense（语义向量）+ Sparse（BM25 关键词）融合
- **智能重排序**：BGE-Reranker 跨编码器精排
- **查询改写**：LLM 驱动的多轮对话查询优化
- **多租户 ACL**：租户隔离 + 访问控制列表
- **用户系统**：首个用户自动成为 admin，固定测试账号可通过环境变量配置专属登录 token
- **异步任务**：上传解析、思维导图、数据表格生成均可返回处理中状态并由后台线程完成
- **思维导图/数据表格**：基于 LLM 的知识结构化生成
- **完整评估框架**：Recall@K、MRR@K、nDCG@K、答案忠实度

---

## 2. 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Frontend (React + Vite)                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐                            │
│  │ Source   │  │  Chat    │  │  Studio   │                            │
│  │ Panel    │  │  Panel   │  │  Panel    │                            │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘                            │
│       │             │              │                                    │
│       └─────────────┼──────────────┘                                    │
│                     │  /api/*  (Vite Proxy → :8008)                     │
└─────────────────────┼──────────────────────────────────────────────────┘
                      │
┌─────────────────────┼──────────────────────────────────────────────────┐
│                     ▼         Backend (FastAPI + Uvicorn)               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        serve.py                                  │   │
│  │  /health  /query  /sources  /conversations  /artifacts           │   │
│  │  /search  /sources/upload  /sources/content  /admin/*            │   │
│  └───┬───────┬──────────┬──────────┬──────────┬────────────────────┘   │
│      │       │          │          │          │                         │
│      ▼       ▼          ▼          ▼          ▼                         │
│  ┌──────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌──────────┐                    │
│  │Query │ │Search│ │Ingest  │ │Conv  │ │Artifact  │                    │
│  │Rewrite│ │Hybrid│ │Pipeline│ │Mgr   │ │Generator │                    │
│  └──┬───┘ └──┬───┘ └───┬────┘ └──┬───┘ └────┬─────┘                    │
│     │        │         │         │          │                            │
│     │   ┌────┼─────────┼─────────┼──────────┼──────┐                    │
│     │   │    │         │         │          │      │                    │
│     ▼   ▼    ▼         ▼         ▼          ▼      │                    │
│  ┌──────────────────────────────────────────────┐  │                    │
│  │              LLM Gateway                      │  │                    │
│  │  SiliconFlow API / OpenAI-compatible          │  │                    │
│  │  • Embedding: BAAI/bge-m3                     │  │                    │
│  │  • Reranker: BAAI/bge-reranker-v2-m3          │  │                    │
│  │  • Chat: DeepSeek-V4-Flash / Gemini-3-Flash   │  │                    │
│  │  • Vision: Qwen/Qwen3-VL-8B-Instruct          │  │                    │
│  └──────────────────────────────────────────────┘  │                    │
│                                                     │                    │
│  ┌──────────────────────────────────────────────┐  │                    │
│  │            Milvus (Vector Database)           │  │                    │
│  │  • Collection: rag_chunks_v1                  │  │                    │
│  │  • Dense Index: HNSW/COSINE                   │  │                    │
│  │  • Sparse Index: BM25 (built-in function)      │  │                    │
│  │  • Image Index: HNSW/COSINE                   │  │                    │
│  └──────────────────────────────────────────────┘  │                    │
│                                                     │                    │
│  ┌──────────────────────────────────────────────┐  │                    │
│  │        Object Store (Filesystem)              │  │                    │
│  │  • canonical/source_documents.jsonl           │  │                    │
│  │  • canonical/source_guides.jsonl              │  │                    │
│  │  • current_versions.json                      │  │                    │
│  │  • artifacts/<tenant>/                        │  │                    │
│  └──────────────────────────────────────────────┘  │                    │
│                                                     │                    │
│  ┌──────────────────────────────────────────────┐  │                    │
│  │       Metadata DB (SQLite)                    │  │                    │
│  │  • users, sessions, announcements             │  │                    │
│  │  • conversations, messages                    │  │                    │
│  │  • artifacts, source_tasks                    │  │                    │
│  └──────────────────────────────────────────────┘  │                    │
└─────────────────────────────────────────────────────┘                    │
```

---

## 3. 部署架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                          │
│                                                                   │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐                          │
│  │  etcd   │  │  MinIO  │  │ Milvus   │  ← 基础设施层             │
│  │ v3.5.18 │  │ 2024-05 │  │ v2.6.13  │                          │
│  └────┬────┘  └────┬────┘  └────┬─────┘                          │
│       │            │            │                                  │
│       └────────────┼────────────┘                                  │
│                    │                                               │
│  ┌─────────────────┼──────────────────────────────────────────┐   │
│  │            rag-api (FastAPI :8008)                          │   │
│  │  • Dockerfile 构建                                          │   │
│  │  • healthcheck: /health                                     │   │
│  │  • 挂载: runtime/, object_store/                            │   │
│  └─────────────────┼──────────────────────────────────────────┘   │
│                    │                                               │
│  ┌─────────────────┼──────────────────────────────────────────┐   │
│  │            rag-web (Nginx :8080 → React SPA)                │   │
│  │  • 反向代理 API 请求到 rag-api                               │   │
│  │  • 静态文件服务                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │      rag-ingest (profile: ingest, on-demand)                │   │
│  │  • 批量摄入脚本，启动后自动退出                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**外部 LLM 服务**：所有 Embedding、Rerank、Chat 调用均通过 SiliconFlow API 网关（`https://api.siliconflow.cn`），无需本地 GPU。

---

## 4. 数据流全景

```
用户上传 PDF/MD/HTML/TXT/CSV
         │
         ▼
   ┌─────────────┐
   │ 文件解析     │  PyMuPDF / Markdown Splitter / CSV Reader
   │ + PII 检测   │  检测邮箱、手机号、身份证号等敏感信息
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ 文档分块     │  结构化分块 (保留代码块/表格完整性)
   │ Chunking     │  chunk_size=700 tokens, overlap=100
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ 向量嵌入     │  text → BAAI/bge-m3 (1024d)
   │ Embedding    │  image → Qwen3-VL-Embedding-8B (1024d)
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ 写入 Milvus  │  HNSW (dense) + BM25 (sparse) + HNSW (image)
   │ + 归档到     │  object_store/canonical/source_documents.jsonl
   │ Object Store │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ 版本发布     │  current_versions.json
   │ + 源指南生成 │  LLM 生成文档摘要，供来源解读和查询改写使用
   └─────────────┘


用户查询流程：

   "Transformer 注意力机制如何工作？"
          │
          ▼
   ┌──────────────┐
   │ 1. 查询改写   │  LLM: 文档摘要 + 多轮对话上下文 → 检索优化查询
   │   Rewrite     │  "Transformer 注意力机制 自注意力 多头注意力 原理"
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ 2. 向量编码   │  BAAI/bge-m3 → 1024维 语义向量
   │   Embed       │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ 3. 混合检索   │  Dense (HNSW/COSINE) + Sparse (BM25) → RRF 融合
   │   Hybrid      │  k=60, candidate_limit=20
   │   Search      │  + ACL/tenant/doc_version 过滤
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ 4. 重排序     │  BGE-Reranker-v2-m3 跨编码器
   │   Rerank      │  对所有候选文档精排
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ 5. 上下文打包 │  • min_rerank_score 过滤
   │   Packing     │  • max 5 chunks
   │               │  • 每篇文档最多 2 个chunk
   │               │  • 总字符预算 6000
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ 6. 答案生成   │  LLM + 证据上下文 → 结构化回答
   │   Generate    │  含引用标注 + 图片展示
   └──────────────┘
```

---

## 5. 文档摄取管道

### 5.1 支持的文件格式

| 格式 | 解析引擎 | 特殊处理 |
|------|---------|---------|
| **PDF** | PyMuPDF (fitz) | 文本块 + 表格 + 嵌入图片提取 + 图片描述生成 |
| **Markdown** | 自定义 Section Splitter | 按标题层级分段 |
| **HTML** | 自定义 VisibleTextParser | 去标签、提取纯文本 + 标题路径 |
| **TXT** | 直接读取 | — |
| **CSV/TSV** | csv 模块 | 转 Markdown 表格，大表按 200 行拆分 |

### 5.2 PDF 解析详解（PyMuPDF）

```
extract_pdf_pages_with_pymupdf()
│
├── 逐页处理
│   ├── extract_pymupdf_text_blocks()    → 文本块
│   ├── extract_pymupdf_tables()         → 表格 → Markdown
│   └── 嵌入图片处理
│       ├── 提取图片 → SHA256 缓存到 assets/ 目录
│       ├── PdfImageCaptioner
│       │   └── Qwen/Qwen3-VL-8B-Instruct (视觉模型)
│       │       max_tokens=256, 中日文自动检测
│       │       RAG_PDF_CAPTION_MAX_IMAGES=24 (每 PDF 最多描述数)
│       └── 图片描述注入到检索文本中
│
├── 输出: PdfPage
│   ├── text: 检索用文本 (含图片描述)
│   ├── display_text: 展示用文本 (纯文本)
│   └── display_blocks: [{"type":"image", "title":"...", "url":"data:image/..."}]
│
└── 每页 → 一个 SourceDocument (doc_id = <path>/page-<N>)
```

### 5.3 分块策略

```python
# rag_core/text_utils.py

chunk_document(doc, chunk_size=700, overlap=100)
│
├── split_structural_blocks(text)
│   • 保留 ``` 代码块完整性
│   • 保留 Markdown 表格完整性
│   • 按段落边界切分
│
├── 块累积逻辑
│   • 逐块累加 token 计数
│   • 超出 chunk_size → flush 当前 chunk
│   • 超大块: token 滑动窗口切分
│
└── make_chunk(doc, chunk_index, body)
    → Chunk 文本格式:
      "标题路径: {title}
       来源: {source_type}
       正文:
       {body.strip()}"
```

### 5.4 PII 检测

系统在摄入阶段对每个文档文本运行 PII 检测（`rag_core/pii.py`）：

| 策略 | 行为 |
|------|------|
| `warn` (默认) | 打印警告日志，不修改内容 |
| `redact` | 替换敏感信息为 `[REDACTED]` |
| `fail` | 拒绝包含 PII 的文档 |

检测模式：邮箱、中国手机号、中国身份证号、API Key/Token。

### 5.5 内容寻址与去重

```
doc_id = "filename@sha256-abc123def456"

SHA256 摘要基于:
  • tenant_id + source_uri + 内容哈希
  → 同名同内容文件 → 相同 doc_id
  → 自动版本管理: next_source_doc_version()
```

### 5.6 异步上传任务

前端上传文件后，`/sources/upload` 先把任务写入 SQLite `source_tasks` 表并返回 `status="processing"`。后台线程继续执行解析、PII 检测、分块、向量化、Milvus 写入、对象存储归档和源指南生成。

```
POST /sources/upload
  ├── 保存原始文件到 object_store/uploads/<tenant>/<uuid>/
  ├── create_source_task(status="processing")
  ├── BackgroundThread: ingest_upload_background()
  │   ├── ingest_uploaded_file()
  │   ├── publish_current_versions()
  │   ├── get_or_create_source_guide()
  │   └── update_source_task(status="ready" | "failed")
  └── 前端轮询 /sources，展示 processing / failed / ready 状态
```

---

## 6. 多模态处理

### 6.1 PDF 图片提取与描述

```
PDF 页面
  │
  ├── 提取文本块 + 表格 (检索用)
  │
  └── 嵌入图片
      ├── 保存到 {pdf}.assets/page-{N}-image-{M}-{sha256}.png
      │   (SHA256 摘要缓存，避免重复提取)
      │
      └── 生成图片描述 (可选)
          ├── 后端: siliconflow (Qwen/Qwen3-VL-8B-Instruct)
          ├── 语言检测: CJK 字符占比 > 50% → 中文 prompt
          └── 描述注入到检索文本:
              "Image 1 caption: 图中展示了 Transformer 的注意力机制架构图..."
```

### 6.2 图片向量化

每个嵌入图片产生两个向量：

| 向量类型 | 编码器 | 用途 |
|---------|--------|------|
| **文本向量** | BAAI/bge-m3 (图片描述文本) | 语义检索 |
| **图片向量** | Qwen3-VL-Embedding-8B (图片文件) | 视觉相似度检索 |

### 6.3 多模态检索流程

```
search_multimodal.py → retrieve_multimodal()

文本查询模式:
  ┌─────────────┐     ┌──────────────┐
  │ Query Rewrite│ ──► │ Text Embed   │
  └─────────────┘     └──────┬───────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
    ┌──────────────────┐          ┌──────────────────┐
    │ Hybrid Search     │          │ Image Vector     │
    │ (text_dense + BM25)│          │ Search (HNSW)    │
    │ source_types=image │          │ image_dense_vector│
    └────────┬─────────┘          └────────┬─────────┘
              │                            │
              └──────────┬─────────────────┘
                         ▼
              ┌──────────────────┐
              │ RRF Fusion (k=60) │
              └────────┬─────────┘
                         ▼
              ┌──────────────────┐
              │ Context Packing   │
              │ (无 Reranker)     │
              └────────┬─────────┘
                         ▼
              ┌──────────────────┐
              │ Answer Generate   │
              └──────────────────┘

图片查询模式 (用户上传图片):
  跳过 Text Hybrid Search，仅做 Image Vector Search
```

---

## 7. 检索管道

### 7.1 全文检索 (retrieve_and_rerank)

```
rag_core/pipeline.py

┌─────────────────────────────────────────────────────────┐
│ 1. Query Rewrite (LLM)                                  │
│    • 源指南摘要 source_guides.jsonl (最多 20 条)          │
│    • 多轮对话历史 (6 轮)                                 │
│    • 输出: 检索优化后的短查询                              │
│    • 租户越界检测: 改写结果涉及其他租户 → 空结果           │
├─────────────────────────────────────────────────────────┤
│ 2. Dense Embedding                                      │
│    • BAAI/bge-m3 → 1024 维向量                          │
├─────────────────────────────────────────────────────────┤
│ 3. Hybrid Search (Milvus)                               │
│    • 并行请求:                                           │
│      - Dense:  text_dense_vector → HNSW/COSINE (ef=128) │
│      - Sparse: bm25_sparse_vector → BM25                │
│    • Filter Expression:                                  │
│      - tenant_id == "tenant-xxx"                         │
│      - is_active == true                                 │
│      - ARRAY_CONTAINS_ANY(acl_groups, [...])             │
│      - source_type IN [...]  (可选)                      │
│    • RRF Fusion: k=60                                    │
│    • candidate_limit=20                                  │
├─────────────────────────────────────────────────────────┤
│ 4. Rerank                                               │
│    • BGE-Reranker-v2-m3 (全量候选)                       │
│    • 按 rerank_score 降序排列                             │
├─────────────────────────────────────────────────────────┤
│ 5. Context Packing                                       │
│    • min_rerank_score, max_chunks_per_doc=2,             │
│      context_limit=5, max_context_chars=6000            │
├─────────────────────────────────────────────────────────┤
│ 6. Answer Generation                                     │
│    • 证据上下文 → LLM → 结构化回答                        │
└─────────────────────────────────────────────────────────┘
```

### 7.2 三种检索模式

| 模式 | 搜索字段 | 融合方式 | 适用场景 |
|------|---------|---------|---------|
| **Dense** | text_dense_vector (HNSW) | — | 纯语义检索 |
| **Sparse** | bm25_sparse_vector (BM25) | — | 关键词精确匹配 |
| **Hybrid** | Dense + Sparse | RRF (k=60) | 语义+关键词综合（**默认**） |
| **Multimodal** | text_hybrid + image_dense_vector | RRF (k=60) | 图文混合检索 |

### 7.3 Milvus Filter Expression 构建

```python
# rag_core/milvus_store.py: build_filter_expr()

filter = " and ".join([
    f'tenant_id == "{tenant_id}"',          # 多租户隔离
    "is_active == true",                    # 软删除排除
    f'ARRAY_CONTAINS_ANY(acl_groups, [...])', # ACL 权限
    f"doc_version {version_clause}",         # 版本过滤
    f'source_type in [...]',                # 类型过滤 (可选)
    f'doc_id in [...]',                     # 文档过滤 (可选)
    f'embedding_model == "{model}"',        # 模型一致性
])
```

### 7.4 查询改写上下文来源

查询改写的用户 prompt 由三段组成：

```
资料摘要:
  来自 object_store/canonical/source_guides.jsonl
  按 tenant_id、doc_id、doc_version/current_versions 过滤

对话历史:
  最近 RAG_QUERY_REWRITE_HISTORY_TURNS 轮历史

当前问题:
  用户本轮输入
```

当用户指定的 `doc_ids` 是 PDF 页、图片等子文档 id，系统会先精确匹配源指南；若没有命中，则回退到当前租户当前版本的源指南摘要，避免 PDF 父文档摘要无法参与改写。

---

## 8. 重排序

### 8.1 后端选择

| 后端 | 模型 | 实现 | 特点 |
|------|------|------|------|
| **siliconflow** | BAAI/bge-reranker-v2-m3 | API 调用 `/v1/rerank` | 无需 GPU，低延迟 |
| **bge** | BAAI/bge-reranker-v2-m3 | 本地加载 Transformers | 离线可用，需 GPU |

### 8.2 本地 BGE 重排序详解

```
TransformersBGEReranker (rag_core/rerankers.py)
│
├── 模型: AutoModelForSequenceClassification
│   from BAAI/bge-reranker-v2-m3
│
├── 推理: (query, document_text) 对 → logit 分数
│   • batch_size = RAG_RERANK_BATCH_SIZE (default 8)
│   • max_length = RAG_RERANK_MAX_LENGTH (default 1024)
│   • 截断策略: 仅截断文档侧 (query: 前 512 tokens)
│
└── 输出: hit.rerank_score (类似 relevance score)
```

---

## 9. 上下文打包与答案生成

### 9.1 上下文打包策略

```python
# rag_core/context.py: pack_context()

上下文打包三重约束:

1. 最低重排序分  (RAG_MIN_RERANK_SCORE, 可选)
   └── 过滤掉低相关性文档

2. 每篇文档上限  (RAG_MAX_CHUNKS_PER_DOC = 2)
   └── 防止单篇文档占据全部上下文

3. 字符预算      (RAG_MAX_CONTEXT_CHARS = 6000)
   └── 累积文本字符数到达上限即停止

4. 数量上限      (context_limit = 5)
   └── 最多选取 5 个 chunk
```

### 9.2 答案生成 Prompt 结构

```
SYSTEM PROMPT:
  你是企业知识库问答助手。
  你的任务是优先基于检索证据回答...

USER PROMPT:
  当前日期: {today}
  检索证据:
  [1] doc_id=... title=... source_type=... source_uri=... chunk_index=...
  {chunk_text}

  [2] doc_id=... ...

  回答策略:
  1. 优先使用证据中的信息
  2. 标注引用来源 [N]
  3. 证据不充分时说明"当前知识库没有足够证据"
  ...

  用户问题: {query}
```

### 9.3 引用系统

- 每个证据块在 prompt 中以 `[N]` 编号
- LLM 生成的答案中使用 `[N]` 引用对应证据
- 前端解析 `[N]` 并渲染为可点击的引用卡片
- 引用卡片展示: 文档标题、来源类型、页码、相关图片

---

## 10. Milvus 模式设计

### 10.1 Collection: rag_chunks_v1

```
┌──────────────────────┬──────────────────────┬─────────────────────────┐
│ 字段名                │ 类型                  │ 说明                     │
├──────────────────────┼──────────────────────┼─────────────────────────┤
│ id                   │ VARCHAR(128) PK       │ tenant:doc:version:chunk │
│ tenant_id            │ VARCHAR(64)           │ 多租户隔离                │
│ doc_id               │ VARCHAR(128)          │ 文档标识                  │
│ doc_version          │ INT64                 │ 版本号                    │
│ chunk_index          │ INT64                 │ 块序号                    │
│ source_type          │ VARCHAR(32)           │ md/pdf/html/image/table   │
│ source_uri           │ VARCHAR(512)          │ 原始文件路径               │
│ title                │ VARCHAR(512)          │ 标题路径                   │
│ text                 │ VARCHAR(8192)         │ 块文本 (enable_analyzer)  │
│ language             │ VARCHAR(16)           │ zh/en/auto               │
│ acl_groups           │ ARRAY(VARCHAR)        │ ACL 组 (max 32 × 64)     │
│ created_at           │ INT64                 │ 创建时间戳 (ms)            │
│ updated_at           │ INT64                 │ 更新时间戳 (ms)            │
│ is_active            │ BOOL                  │ 软删除标记                │
│ embedding_model      │ VARCHAR(128)          │ 嵌入模型名                │
│ embedding_dim        │ INT64                 │ 向量维度                  │
│ content_hash         │ VARCHAR(64)           │ 文本 SHA256              │
│ text_dense_vector    │ FLOAT_VECTOR(1024)    │ 语义向量                  │
│ bm25_sparse_vector   │ SPARSE_FLOAT_VECTOR   │ BM25 向量 (自动生成)       │
│ image_dense_vector   │ FLOAT_VECTOR(1024)    │ 图片向量                  │
│ metadata             │ JSON                  │ 页码/bbox/row_range 等    │
└──────────────────────┴──────────────────────┴─────────────────────────┘
```

### 10.2 索引配置

| 索引 | 类型 | 参数 | 说明 |
|------|------|------|------|
| `text_dense_vector` | HNSW | M=16, efConstruction=100, COSINE | 语义检索 |
| `bm25_sparse_vector` | SPARSE_INVERTED_INDEX | drop_ratio_build=0.2 | 关键词检索 |
| `image_dense_vector` | HNSW | M=16, efConstruction=100, COSINE | 图片检索 |

### 10.3 BM25 自动生成函数

```
text_bm25_function:
  FunctionType.BM25
  input_field_names: ["text"]
  output_field_names: ["bm25_sparse_vector"]
  → Milvus 在 insert/upsert 时自动计算 BM25 稀疏向量
```

---

## 11. LLM 调用全景图

### 11.1 所有 LLM 调用点一览

| # | 调用位置 | 文件 | 模型 (config key) | 用途 |
|---|---------|------|-------------------|------|
| 1 | 查询改写 | `rag_core/rewrite.py` | `llm_model` | 多轮对话查询优化 |
| 2 | 答案生成 | `rag_core/answering.py` | `llm_model` | 基于证据生成回答 |
| 3 | 图片描述 | `rag_core/io.py:623-679` | `RAG_PDF_IMAGE_CAPTION_MODEL` | PDF 图片视觉描述 |
| 4 | 源指南 | `rag_core/source_guides.py` | `llm_model` | 文档内容摘要生成 |
| 5 | 思维导图 | `rag_core/artifacts.py:344` | `llm_model` | 部分导图生成 (JSON) |
| 6 | 合并/表格 | `rag_core/artifacts.py:359` | `llm_model` | 导图合并+表格生成 (JSON) |

### 11.2 LLM 配置

```bash
# .env 配置
NEW_API_URL="https://api.siliconflow.cn"
NEW_API_KEY="sk-..."
LLM_MODEL="deepseek-ai/DeepSeek-V4-Flash"   # Docker 默认
# LLM_MODEL="gemini-3-flash-preview"         # 本地默认
```

`NEW_API_URL` 会在 `rag_core/config.py` 中自动补齐 `/v1` 后缀，用于查询改写、答案生成、源指南、Studio 思维导图和表格生成。

### 11.3 非 LLM 的模型 API 调用

| 服务 | 模型 | API 端点 | 用途 |
|------|------|---------|------|
| 文本嵌入 | BAAI/bge-m3 | `/v1/embeddings` | 1024 维语义向量 |
| 图片嵌入 | Qwen3-VL-Embedding-8B | `/v1/embeddings` | 1024 维视觉向量 |
| 重排序 | BAAI/bge-reranker-v2-m3 | `/v1/rerank` | 相关性精排 |
| 图片描述 | Qwen/Qwen3-VL-8B-Instruct | `/v1/chat/completions` | Vision LLM 描述 |

### 11.4 本地模型后备

当 `RAG_EMBEDDING_BACKEND=bge` 或 `RAG_RERANK_BACKEND=bge` 时，系统从 ModelScope 加载本地模型：

```
~/.cache/modelscope/hub/models/BAAI/
├── bge-m3/                    # 文本嵌入 (Transformers)
│   └── onnx/                  # ONNX 加速
├── bge-reranker-v2-m3/        # 交叉编码器重排序
└── ...
```

---

## 12. 对象存储与版本管理

### 12.1 目录结构

```
object_store/
├── uploads/
│   └── <tenant>/<uuid>/<safe_filename>    # 原始上传文件
├── canonical/
│   ├── source_documents.jsonl              # 所有版本的文档归档
│   ├── deleted_documents.jsonl             # 删除墓碑记录
│   ├── source_guides.jsonl                  # LLM 生成的文档摘要
│   └── source_section_summaries.jsonl       # 章节级提取摘要
├── current_versions.json                   # {tenant: {doc_id: version}}
└── artifacts/
    └── <tenant>/
        └── <artifact_id>.json              # 思维导图/表格产物
```

### 12.2 版本管理

```python
# rag_core/versioning.py

publish_current_versions(object_store_dir, docs)
  → 写入 current_versions.json
  → {tenant_id: {doc_id: version}}

load_current_versions(object_store_dir, tenant_id)
  → 检索时获取每个文档的活跃版本

# 查询时支持指定 doc_version 参数
# 指定版本 → 历史版本查询
# 未指定版本 → 使用 current_versions (活跃版本)
```

### 12.3 删除机制

- **软删除**: source_tasks 记录标记为 `deleted`
- **墓碑**: `deleted_documents.jsonl` 记录已删除的 doc_id
- **Milvus**: `is_active = false` (不物理删除向量)
- **重建支持**: `rebuild_from_object_store.py --reset` 从归档重建

### 12.4 源指南摘要

`canonical/source_guides.jsonl` 保存每个来源文档当前版本的 LLM 摘要：

```
{
  "tenant_id": "tenant-xxx",
  "source_doc_id": "自然辩证法.pdf@sha256-...",
  "doc_version": 3,
  "title": "自然辩证法",
  "guide": "这份资料讨论...",
  "model": "deepseek-ai/DeepSeek-V4-Flash"
}
```

用途：

- 来源面板展示文档解读
- 查询改写阶段提供“资料摘要”
- 对 PDF 页、图片等子文档查询提供父文档语义背景
- 重建对象存储时恢复摘要缓存，避免重复 LLM 调用

### 12.5 章节级提取摘要

`canonical/source_section_summaries.jsonl` 按租户、来源文档、版本和章节序号保存确定性提取摘要。
比较、综合、信息抽取和报告任务会把这些摘要作为独立证据加入上下文预算；普通文档总结仍优先使用更紧凑的源指南。
该层不增加模型调用，支持本地与 S3/MinIO 后端，并在删除来源时同步清理。

---

## 13. 认证与授权

### 13.1 三层安全模型

```
Layer 1: API Token (可选)
  RAG_API_TOKEN → Authorization: Bearer <token>
  └── 服务间通信保护

Layer 2: 多租户 ACL
  X-RAG-Tenant-ID + X-RAG-ACL-Groups 请求头
  或 body.tenant_id + body.acl_groups
  └── Milvus filter: tenant_id + ARRAY_CONTAINS_ANY(acl_groups)

Layer 3: 用户认证 (Session)
  PBKDF2-SHA256 密码哈希 (240K 迭代)
  7 天 Session TTL
  └── 自动分配 tenant_id: tenant-<uuid12>
```

### 13.2 用户角色

| 角色 | 权限 |
|------|------|
| **admin** | 用户管理、封禁/解封、发布公告、开关注册 |
| **user** | 上传文档、查询、对话、生成 Artifact |
| **guest** (未登录) | 查看公告（只读） |

### 13.3 注册控制

- 首个注册用户自动成为 admin
- Admin 可通过 API 关闭新用户注册
- 禁用注册后普通注册会被拒绝；已存在用户仍可登录

### 13.4 固定测试账号

首次启动 FastAPI 应用时，`create_app()` 会调用 `ensure_default_test_account()` 自动创建一个普通测试账号：

| 字段 | 值 |
|------|----|
| username | `test_user` |
| password | `12345678` |
| role | `user` |
| tenant_id | `tenant-fixed-test` |
| 默认专属 token | `production-rag-fixed-test-login-token` |

专属 token 可通过 `RAG_FIXED_TEST_LOGIN_TOKEN` 覆盖。真实部署中可把它设置为短而稳定的分享 token，例如：

```bash
RAG_FIXED_TEST_LOGIN_TOKEN=production-rag-fixed-test-login-token
```

该账号的专属 session 采用固定 token 和远期过期时间；调用 `/auth/logout` 时不会删除这个固定 token，因此可用于稳定的体验链接：

```
http://localhost:5173/#token=<RAG_FIXED_TEST_LOGIN_TOKEN>
```

如果管理员把 `test_user` 封禁，`authenticate_token()` 仍会拒绝该 token。

### 13.5 管理员控制台能力

管理员接口覆盖：

- 用户列表、搜索、分页
- 单用户封禁/解封
- 批量更新用户状态、昵称修改权限、头像修改权限
- 注册开关
- 公告发布和删除

这些数据都存储在 SQLite metadata DB 中；用户、sessions、公告、对话、消息、artifact 和 source task 共享同一套 WAL 模式数据库。

---

## 14. 对话管理

### 14.1 数据模型

```
Conversation {
  id: string
  tenant_id: string
  title: string              # 自动从第一条用户消息截取 (40 字)
  messages: [Message]        # 有序消息列表
  source_doc_ids: string[]   # 对话关联的文档
}

Message {
  id: string
  role: "user" | "assistant"
  content: string
  status: "done" | "sending" | "failed"
  request_id: string         # 关联查询请求 (用于反馈)
  citations: Citation[]      # 检索引用
  imageDataUrl: string?      # 用户上传的图片
  feedbackRating: 1 | -1?    # 用户反馈
}
```

### 14.2 持久化

- **主存储**: SQLite metadata DB (`conversations` + `messages` 表)
- **迁移支持**: 从 JSON 文件 (`runtime/conversations/`) 自动迁移到 SQLite
- **租户隔离**: 所有查询带 `tenant_id` 过滤

---

## 15. Studio：思维导图与数据表格

### 15.1 思维导图生成

```
用户选择来源文档 → createMindMap()

Stage 1: 分批生成局部导图
  ├── 每 5 个文档一批
  ├── LLM + PARTIAL_MINDMAP_SYSTEM_PROMPT
  └── 输出: 局部 JSON 树 (id, label, children, citationIds)

Stage 2: 合并导图
  ├── LLM + MERGE_MINDMAP_SYSTEM_PROMPT
  └── 输入: 所有局部导图的 JSON → 统一 JSON 树

Stage 3: 归一化
  ├── 第二层节点: 最多 8 个
  ├── 第三层节点: 最多 6 个
  └── 标签: 截断到 80 字符
```

### 15.2 数据表格生成

```
用户选择来源 → createDataTable()

Stage 1: 拼接文档文本
  ├── 最多 9000 字符
  └── LLM + DATA_TABLE_SYSTEM_PROMPT

Stage 2: 归一化
  ├── 最多 24 行 × 8 列
  └── 单元格截断到 260 字符

输出 Schema:
  {"title": "...", "columns": [...], "rows": [[...]], "summary": "..."}
```

### 15.3 Artifact 存储

- **文件**: `object_store/artifacts/<tenant>/<artifact_id>.json`
- **SQLite**: `artifacts` 表 (镜像)
- **状态**: `ready` | `failed` | `generating`
- **生成方式**: `/artifacts/mindmap` 和 `/artifacts/table` 先创建 pending artifact，再由后台线程调用 LLM；前端通过 `/artifacts/{artifact_id}` 或列表轮询状态

---

## 16. 评估框架与发布门禁

项目测评记录见 [PROJECT_EVALUATION.md](/project-evaluation)，其中包含 retrieval recall、后续压测等项目级结果。

### 16.1 检索评估 (eval_retrieval.py)

```
输入: JSONL 文件
  {"query": "...", "tenant_id": "...", "expected_doc_ids": [...]}

评估指标:
┌─────────────────┬───────────────────────────────────┐
│ 指标              │ 说明                              │
├─────────────────┼───────────────────────────────────┤
│ Recall@K         │ top-K 中命中期望文档的比例          │
│ MRR@K            │ 首个相关文档排名的倒数均值           │
│ nDCG@K           │ 归一化折损累计增益                   │
│ Permission Leak  │ 跨租户泄露检测                      │
│ Latency (p50/p95)│ 各阶段延迟 (嵌入/搜索/重排/生成)      │
└─────────────────┴───────────────────────────────────┘
```

### 16.2 答案评估 (eval_answer.py)

```
输入: 同上 JSONL + expected_answer_terms + unsupported_answer_terms

评估指标:
┌────────────────────┬───────────────────────────────────┐
│ 指标                 │ 说明                              │
├────────────────────┼───────────────────────────────────┤
│ Citation Accuracy   │ 有效引用编号 / 总引用数             │
│ Evidence Hit Rate   │ 至少命中一个期望文档的查询比例       │
│ Refusal Quality     │ 不可回答问题正确拒绝的比例            │
│ Answer Correctness  │ 期望术语覆盖率                      │
│ Faithfulness        │ 1 - 无证据支持的术语比例             │
└────────────────────┴───────────────────────────────────┘
```

### 16.3 发布门禁 (release_gate.py)

```
release_gate.py
├── 运行检索评估 (text + multimodal)
├── 运行答案评估 (text + multimodal)
├── 对比阈值:
│   • recall@5 ≥ 0.8
│   • citation_accuracy ≥ 0.7
│   • faithfulness ≥ 0.8
│   • ...
└── 任意指标不达标 → exit(1) 阻断发布
```

---

## 17. 配置参考

### 17.1 环境变量一览

#### LLM 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LLM_MODEL` | `gemini-3-flash-preview` | 对话模型 |
| `NEW_API_URL` | — | OpenAI 兼容端点 |
| `NEW_API_KEY` | — | OpenAI 兼容 Key |
| `SILICONFLOW_URL` | `https://api.siliconflow.cn` | SiliconFlow 嵌入/重排/视觉描述端点 |
| `SILICONFLOW_API_KEY` | — | SiliconFlow 嵌入、重排、PDF/用户提问图片描述 Key |

#### 嵌入配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RAG_EMBEDDING_BACKEND` | `siliconflow` | 嵌入后端 (siliconflow/bge) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 文本嵌入模型 |
| `EMBEDDING_DIM` | `1024` | 文本向量维度 |
| `RAG_EMBED_BATCH_SIZE` | `8` | 嵌入批大小 |
| `RAG_EMBED_MAX_LENGTH` | `8192` | 最大 token 长度 |
| `RAG_IMAGE_EMBEDDING_BACKEND` | `none` | 图片嵌入后端 |
| `IMAGE_EMBEDDING_MODEL` | `Qwen/Qwen3-VL-Embedding-8B` | 图片嵌入模型 |
| `IMAGE_EMBEDDING_DIM` | `1024` | 图片向量维度 |

#### 重排序配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RAG_RERANK_BACKEND` | `siliconflow` | 重排后端 (siliconflow/bge) |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排模型 |
| `RAG_RERANK_BATCH_SIZE` | `8` | 重排批大小 |
| `RAG_RERANK_MAX_LENGTH` | `1024` | 重排最大长度 |

#### 检索与上下文配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RAG_CHUNK_SIZE` | `700` | 分块 token 预算 |
| `RAG_CHUNK_OVERLAP` | `100` | 分块重叠 |
| `RAG_MAX_CONTEXT_CHARS` | `6000` | 上下文字符预算 |
| `RAG_MAX_CHUNKS_PER_DOC` | `2` | 每文档最大 chunk 数 |
| `RAG_MIN_RERANK_SCORE` | (未设置) | 最低重排分 |
| `RAG_QUERY_REWRITE_HISTORY_TURNS` | `6` | 查询改写历史轮数 |
| `RAG_QUERY_REWRITE_MAX_TOKENS` | `256` | 改写最大 token 数 |
| `RAG_DENSE_SEARCH_EF` | `128` | HNSW 搜索 ef |

#### Milvus

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RAG_MILVUS_URI` | `production_rag.db` (Lite) | Milvus 连接地址 |
| `RAG_COLLECTION` | `rag_chunks_v1` | 集合名称 |
| `MILVUS_TOKEN` | — | Milvus 认证 Token |

#### 认证

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RAG_API_TOKEN` | (未设置) | API Bearer Token |
| `RAG_REQUIRE_AUTH_CONTEXT` | `false` | 强制要求 ACL 头 |
| `RAG_FIXED_TEST_LOGIN_TOKEN` | `production-rag-fixed-test-login-token` | 固定测试账号 `test_user` 的专属登录 token |

#### PDF 处理

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RAG_PDF_IMAGE_CAPTION_BACKEND` | `siliconflow` | 图片描述后端 |
| `RAG_PDF_IMAGE_CAPTION_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | 视觉描述模型 |
| `RAG_PDF_CAPTION_MAX_IMAGES` | `24` | 每 PDF 最多描述图片数 |
| `RAG_QUERY_IMAGE_CAPTION_BACKEND` | `siliconflow` | 用户提问图片描述后端 |
| `RAG_QUERY_IMAGE_CAPTION_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | 用户提问图片描述模型 |
| `RAG_PII_POLICY` | `warn` | PII 策略 (warn/redact/fail) |

### 17.2 SiliconFlow API 端点

```
https://api.siliconflow.cn
├── /v1/embeddings          # 文本 & 图片向量嵌入
├── /v1/rerank              # 重排序
└── /v1/chat/completions    # LLM 对话 & Vision
```

---

## 18. 开发与生产环境

### 18.1 开发环境（本地热重载）

```bash
# 1. 启动基础设施容器
docker compose up -d milvus

# 2. 本地启动后端 (热重载)
RAG_MILVUS_URI="http://127.0.0.1:19530" \
RAG_OBJECT_STORE_DIR="$(pwd)/object_store" \
RAG_RUNTIME_DIR="$(pwd)/runtime" \
uvicorn serve:app --reload --host 0.0.0.0 --port 8008

# 3. 本地启动前端 (热重载)
cd frontend && npm run dev -- --host 0.0.0.0
```

### 18.2 生产环境（Docker Compose）

```bash
# 完整部署
docker compose up -d

# 服务:
#   rag-milvus  → Milvus 向量数据库 (:19530)
#   rag-etcd    → Milvus 元数据存储
#   rag-minio   → Milvus 对象存储
#   rag-api     → FastAPI 后端 (:8008)
#   rag-web     → Nginx + React 前端 (:8080)

# 批量摄入
RAG_TEXT_INPUT="/data/docs" \
RAG_IMAGE_INPUT="/data/images" \
docker compose --profile ingest up rag-ingest
```

### 18.3 API 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/ready` | 依赖就绪检查 |
| POST | `/query` | RAG 查询 (核心接口) |
| POST | `/search` | 仅检索 (不生成回答) |
| GET | `/sources` | 列出来源文档 |
| POST | `/sources/upload` | 上传文档 |
| GET | `/sources/{doc_id}` | 文档详情 |
| GET | `/sources/content/{doc_id}` | 文档内容 (含图片) |
| GET | `/source-assets/{asset_path}` | 来源文档图片资源 |
| PATCH | `/sources/{doc_id}` | 重命名来源文档 |
| DELETE | `/sources/{doc_id}` | 删除文档 |
| GET | `/conversations` | 列出对话 |
| POST | `/conversations` | 创建/更新对话 |
| GET | `/conversations/{id}` | 获取单个对话 |
| DELETE | `/conversations/{id}` | 删除对话 |
| GET | `/artifacts` | 列出 Studio 产物 |
| POST | `/artifacts/mindmap` | 创建思维导图 |
| POST | `/artifacts/table` | 创建数据表格 |
| GET | `/artifacts/{artifact_id}` | 获取 Studio 产物详情 |
| PATCH | `/artifacts/{artifact_id}` | 重命名 Studio 产物 |
| DELETE | `/artifacts/{artifact_id}` | 删除 Studio 产物 |
| POST | `/feedback` | 提交反馈 |
| GET | `/announcements` | 获取公告 |
| POST | `/auth/register` | 用户注册 |
| POST | `/auth/login` | 用户登录 |
| POST | `/auth/logout` | 用户登出 |
| GET/PATCH | `/auth/me` | 个人信息管理 |
| PATCH | `/auth/password` | 修改密码 |
| GET | `/admin/users` | 管理员: 用户列表 |
| PATCH | `/admin/users/{user_id}/status` | 管理员: 封禁/解封用户 |
| PATCH | `/admin/users/bulk` | 管理员: 批量更新用户权限/状态 |
| GET | `/admin/settings` | 管理员: 系统设置 |
| PATCH | `/admin/settings/registration` | 管理员: 注册开关 |
| POST | `/admin/announcements` | 管理员: 发布公告 |
| DELETE | `/admin/announcements/{id}` | 管理员: 删除公告 |

## 模块文件索引

| 模块 | 文件路径 |
|------|---------|
| FastAPI 应用 | `serve.py` |
| 配置管理 | `rag_core/config.py` |
| 检索管道 | `rag_core/pipeline.py` |
| 查询改写 | `rag_core/rewrite.py` |
| 混合检索 | `rag_core/milvus_store.py` |
| 重排序 | `rag_core/rerankers.py` |
| 上下文打包 | `rag_core/context.py` |
| 答案生成 | `rag_core/answering.py` |
| 提示词模板 | `rag_core/prompts.py` |
| 嵌入模型 | `rag_core/embeddings.py` |
| PDF 解析+图片提取 | `rag_core/io.py` |
| 分块工具 | `rag_core/text_utils.py` |
| 多模态检索 | `search_multimodal.py` |
| 文档管理 | `rag_core/sources.py` |
| 对象存储 | `rag_core/object_store.py` |
| 版本管理 | `rag_core/versioning.py` |
| 认证授权 | `rag_core/auth.py`, `rag_core/user_auth.py` |
| 对话管理 | `rag_core/conversations.py` |
| Artifact 生成 | `rag_core/artifacts.py` |
| 源指南 | `rag_core/source_guides.py` |
| PII 检测 | `rag_core/pii.py` |
| 评估框架 | `eval_retrieval.py`, `eval_answer.py` |
| 发布门禁 | `release_gate.py` |
| 事件记录 | `rag_core/events.py` |
| Schema 管理 | `schema.py` |
| 数据模型 | `rag_core/types.py` |
| Milvus 模式 | `rag_core/milvus_store.py:41-110` |
| Docker 部署 | `docker-compose.yml`, `Dockerfile` |
