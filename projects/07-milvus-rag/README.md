# 07-milvus-rag

目标：用 Milvus Lite 跑通 RAG 向量检索的最小闭环，重点训练面试中必须会讲的 collection、schema、insert、search、metadata filter、topK 和评估。

## 运行

先激活虚拟环境：

```bash
source .venv/bin/activate
```

安装依赖后运行：

```bash
python projects/07-milvus-rag/milvus_lite_rag_demo.py
```

如果是单独安装依赖，Milvus Lite 需要 extra：

```bash
uv add "pymilvus[milvus_lite]"
```

脚本会：

1. 创建本地 Milvus Lite 数据库文件 `projects/07-milvus-rag/milvus_lite_demo.db`。
2. 创建 `rag_chunks_demo` collection。
3. 写入一组模拟知识库 chunk。
4. 执行普通向量检索。
5. 执行带 `tenant_id` 和 `source` 的 filtered search。
6. 计算一个小型 `recall@3`。

## 你要观察什么

- collection 的主键、向量字段和 metadata 字段。
- 同一个 query 在不同 filter 下返回结果如何变化。
- topK 变大时 recall 是否提高。
- 检索结果中的 `distance` / `score` 如何排序。

## 参考答案

### 1. collection 的字段如何理解？

这个 demo 的 collection 是 `rag_chunks_demo`，向量维度是 64，metric 是 `COSINE`。每条 entity 表示一个知识库 chunk，主要字段包括：

- `id`：主键，用于唯一定位一条 chunk。
- `vector`：chunk 的 embedding，用于相似度检索。
- `text`：原始 chunk 文本，用于展示检索结果，也可以直接进入 prompt。
- `doc_id`：chunk 所属文档，方便溯源、删除或更新整篇文档。
- `tenant_id`：租户字段，用于权限隔离。
- `source`：文档来源，例如 `handbook` 或 `runbook`，用于按知识来源过滤。

面试中可以说：向量字段负责语义召回，metadata 字段负责权限、来源、版本和业务过滤，二者一起决定 RAG 检索结果是否既相关又可用。

### 2. filter 前后结果为什么会变化？

不加 filter 时，Milvus 会在整个 collection 里找最相似的 chunk，结果可能包含其他租户或其他来源的数据。加上：

```text
tenant_id == 'team_a' and source == 'handbook'
```

之后，候选范围只剩 `team_a` 的 `handbook` chunk，`team_b` 或 `runbook` 的结果会被排除。生产 RAG 必须在检索阶段做权限过滤，否则无权限 chunk 可能占据 topK，甚至泄露给后续 prompt。

### 3. topK 变大时 recall 为什么可能提高？

`topK` 越大，返回候选越多，正确 chunk 出现在候选集里的概率通常越高，所以 `recall@k` 可能上升。但 topK 不是越大越好：它会增加向量检索、rerank 和 prompt 拼接成本，也可能引入更多噪声。线上常见做法是先召回较大的 topK，例如 20 或 50，再 rerank 后只取少量 chunk 给 LLM。

### 4. `distance` / `score` 怎么看？

本 demo 使用 `COSINE`，向量在 `hash_embedding` 中做了归一化。输出里的 `distance` 可以理解为相似度分数，通常越大越相似。实际项目中要确认所用 metric 的分数方向：`COSINE` / `IP` 常见是越大越相似，`L2` 距离则通常越小越相似。

## 面试口述版

RAG 中我会把文档切成 chunk，再把每个 chunk 的 embedding 和 metadata 写入 Milvus。查询时把用户问题编码为 query vector，结合租户、权限、文档来源等 metadata filter 做向量检索，得到候选 chunk。线上通常会把 topK 设得比最终上下文数量大，再接 reranker 或规则过滤，最后把高相关 chunk 送进 LLM。

## 注意

这个 demo 使用确定性 hash embedding，是为了不下载模型、稳定演示 Milvus 检索链路。真实 RAG 项目应替换为 `bge`、`e5`、`text-embedding` 等真实 embedding 模型，并用标注集评估 `recall@k`、MRR 和端到端回答质量。
