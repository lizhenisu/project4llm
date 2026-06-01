# Milvus 向量检索与 RAG 面试教程

目标：面试 AI Agent / RAG 岗位时，能把 Milvus 向量检索从“会调用 API”讲到“知道为什么这样设计、如何调参、如何排障”。

本教程按面试常问链路组织：

1. 向量检索解决什么问题。
2. Milvus 的数据模型和写入查询流程。
3. metric、index、topK、过滤条件如何影响召回、延迟和成本。
4. RAG 中如何做 chunk、embedding、metadata、hybrid search、rerank。
5. 常见事故如何定位。

官方文档校准点：

- Milvus Quickstart 使用 `pymilvus.MilvusClient` 和 Milvus Lite，可本地文件方式启动。
- Milvus 支持 `FLAT`、`IVF_FLAT`、`IVF_SQ8`、`IVF_PQ`、`HNSW`、`DISKANN`、`AUTOINDEX` 等向量索引，稠密向量常用 `L2`、`IP`、`COSINE`。
- Milvus 支持带 scalar filter 的 filtered search，也支持多向量 hybrid search 和 `WeightedRanker`、`RRFRanker` rerank。

参考：

- https://milvus.io/docs/quickstart.md
- https://milvus.io/docs/index-explained.md
- https://milvus.io/docs/filtered-search.md
- https://milvus.io/docs/hybridsearch.md
- https://milvus.io/docs/reranking.md

## 1. 先建立面试心智模型

RAG 检索可以拆成 6 步：

```text
原始文档 -> chunk -> embedding -> Milvus 写入 -> query embedding -> topK 检索 -> rerank -> 组 prompt
```

Milvus 主要负责：

- 保存向量和 metadata。
- 根据相似度找 topK 近邻。
- 用 scalar filter 缩小搜索范围。
- 用索引在召回率、延迟、内存、构建成本之间做权衡。
- 在多向量场景中合并多个检索结果。

面试时不要只说“把向量存进 Milvus 然后 search”。更完整的回答是：

> 我会先根据业务文档设计 chunk 粒度和 metadata，把 chunk embedding 写入 Milvus collection。查询时先把问题编码成 query embedding，根据租户、权限、文档类型、时间等 metadata 做过滤，再用合适的 metric 和 index 取 topK。线上一般还会扩大召回 topK 后接 cross-encoder 或 LLM rerank，最后把高置信 chunk 拼进 prompt。评估时看 recall@k、MRR、answer hit rate、延迟和资源成本。

## 2. Milvus 核心概念

### Collection

Collection 类似数据库表，保存一类实体。例如知识库 chunk 表：

```text
kb_chunks
  id: primary key
  dense_vector: FLOAT_VECTOR
  text: VarChar
  doc_id: VarChar
  tenant_id: VarChar
  source: VarChar
  created_at: Int64
```

面试要点：

- collection schema 要围绕查询方式设计，不只是围绕写入方便设计。
- RAG 必须保留原文、文档 ID、权限字段、版本字段，否则后面无法溯源、过滤和增量更新。
- metadata filter 是生产 RAG 的基本能力，不是附加功能。

### Entity

Entity 是 collection 中的一行。RAG 中通常一个 entity 对应一个 chunk，而不是一整篇文档。

为什么不用整篇文档直接 embedding：

- 文档太长会超过 embedding 模型输入长度。
- 长文向量会稀释局部语义。
- 召回后塞进 prompt 的上下文成本太高。

### Vector Field

向量字段保存 embedding。常见是一个 dense vector，也可以有多个向量字段：

- `title_dense`：标题语义。
- `body_dense`：正文语义。
- `sparse_vector`：关键词/稀疏检索。
- `image_dense`：图片向量。

多向量的意义：同一个对象可被不同信号命中，再用 rerank 合并。

### Metric

常见 metric：

| metric | 含义 | 适用 |
| --- | --- | --- |
| `COSINE` | 方向相似度 | 文本 embedding 最常见 |
| `IP` | 内积 | 向量已归一化时等价于 cosine 排序，常用于 dense/sparse |
| `L2` | 欧氏距离 | 图像、传统向量或模型明确要求 |

面试陷阱：

- embedding 模型训练时用什么相似度，检索时最好保持一致。
- 如果用 `IP` 模拟 cosine，需要先归一化向量。
- 分数方向不总相同：cosine/IP 通常越大越相似，L2 距离越小越相似。写阈值逻辑时要确认返回分数语义。

## 3. 索引为什么重要

暴力检索 `FLAT` 会逐个计算 query 和库内向量的距离，召回准确但数据大时慢。

ANN 索引用近似搜索换速度：

```text
更快查询 + 更低资源成本 <-> 可能损失少量召回
```

常见索引：

| 索引 | 直觉 | 适合场景 | 常见参数 |
| --- | --- | --- | --- |
| `FLAT` | 全量精确扫描 | 小数据、评估基线 | 无 |
| `IVF_FLAT` | 先聚类到倒排桶，再查部分桶 | 百万级常见基线 | `nlist`、`nprobe` |
| `IVF_SQ8` | IVF 加标量量化 | 内存更紧 | `nlist`、`nprobe` |
| `IVF_PQ` | IVF 加乘积量化 | 大规模、可接受精度损失 | `m`、`nbits` |
| `HNSW` | 图搜索 | 低延迟高召回，内存更高 | `M`、`efConstruction`、`ef` |
| `DISKANN` | 磁盘 ANN | 超大规模、内存受限 | 依部署配置 |
| `AUTOINDEX` | 让 Milvus 选择 | 托管/快速起步 | 依服务实现 |

### IVF 怎么讲

IVF 先把向量空间聚成 `nlist` 个簇。查询时只搜索最接近 query 的 `nprobe` 个簇。

- `nlist` 大：桶更细，构建成本更高。
- `nprobe` 大：查更多桶，召回更高但更慢。
- `nprobe` 小：更快但可能漏掉正确 chunk。

面试口述：

> IVF 的核心是先粗召回再精排距离。调参时我会以 FLAT 结果作为近似真值，扫 `nprobe` 看 recall@k 和 p95 latency 的曲线，而不是只凭经验取固定值。

### HNSW 怎么讲

HNSW 把向量组织成多层近邻图，查询时从高层快速跳到近邻区域，再到底层精细搜索。

- `M` 越大，每个点连接更多邻居，召回更好，内存更高。
- `efConstruction` 越大，构建更慢，图质量更高。
- `ef` 越大，查询时探索更多候选，召回更高，延迟更高。

面试口述：

> HNSW 通常适合对低延迟和召回都要求较高的在线检索，但内存占用明显高于 IVF/PQ。线上调 `ef` 比较灵活，可以按延迟预算动态调整。

## 4. RAG 中的 schema 设计

一个面试可接受的 knowledge chunk schema：

```text
id: int64 primary key
vector: float_vector(dim=embedding_dim)
text: varchar(max_length=4096)
doc_id: varchar
chunk_id: int64
tenant_id: varchar
acl_group: varchar
source: varchar
version: int64
created_at: int64
```

设计理由：

- `doc_id + chunk_id` 用于定位原文。
- `tenant_id / acl_group` 用于权限隔离。
- `source / created_at / version` 用于过滤、更新和回滚。
- `text` 用于直接组 prompt；大字段也可以只存引用，把正文放对象存储。

### 什么时候用 partition

partition 可以把数据物理或逻辑分组，但不要把每个用户、每篇文档都做成 partition。

适合：

- 少量稳定大类，例如业务线、语种、冷热数据。
- 查询天然只落在某几个分区。

不适合：

- 高基数字段，例如 user_id、doc_id。
- 频繁创建和删除的小分区。

高基数字段更常用 scalar field + scalar index + filter。

## 5. 写入链路

写入前要保证：

1. embedding 维度和 collection schema 一致。
2. metric 和向量归一化策略一致。
3. 每个 chunk 有稳定 ID，方便幂等更新。
4. metadata 字段完整，尤其是权限和溯源字段。
5. 批量写入，避免单条频繁 insert。

生产写入常见流程：

```text
解析文档 -> 清洗 -> chunk -> embedding -> 批量 upsert/insert -> build/load index -> smoke search -> 发布版本
```

面试补充：

- 新旧版本切换可以用 `version` 字段过滤，也可以维护两套 collection 做蓝绿发布。
- 删除文档时要按 `doc_id` 删除所有 chunk。
- embedding 模型升级时通常需要重算全量向量，因为新旧向量空间不可直接混用。

## 6. 查询链路

最小查询：

```python
results = client.search(
    collection_name="kb_chunks",
    data=[query_vector],
    limit=5,
    output_fields=["text", "doc_id", "source"],
)
```

带过滤：

```python
results = client.search(
    collection_name="kb_chunks",
    data=[query_vector],
    filter="tenant_id == 'team_a' and source == 'handbook'",
    limit=5,
    output_fields=["text", "doc_id", "source"],
)
```

面试要点：

- filter 通常先缩小候选范围，再做向量检索。
- 权限过滤必须在检索阶段做，不能等返回后再简单丢弃，否则 topK 可能被无权限结果占满。
- topK 不等于最终塞给 LLM 的 chunk 数。常见做法是检索 topK=20/50，rerank 后取 3-8 个。

## 7. Hybrid Search 与 Rerank

为什么 dense search 不够：

- 专有名词、订单号、错误码、函数名等需要精确匹配。
- 用户 query 很短时语义向量可能不稳定。
- 多模态或多字段对象不能只靠一个向量。

常见方案：

```text
dense vector search + sparse/BM25 search + metadata filter + rerank
```

Milvus 支持多向量 hybrid search：对多个向量字段分别发起 ANN search，再用 `WeightedRanker` 或 `RRFRanker` 合并。

两种 rerank 思路：

- `WeightedRanker`：不同检索通道按权重融合，适合你知道 dense/sparse 哪个更重要。
- `RRFRanker`：按排名倒数融合，适合不同通道分数不可直接比较。

工程上更常见的两段式：

1. Milvus 召回较大的候选集。
2. 用 cross-encoder、LLM reranker 或业务规则 rerank。

面试口述：

> 我不会指望向量库一次 search 就解决所有相关性问题。Milvus 负责高效召回，reranker 负责更精细的 query-document 交互判断。召回阶段宁可多取一些候选，避免正确 chunk 过早丢失。

## 8. Chunk 策略

chunk 是 RAG 质量的上游关键。

常见策略：

| 策略 | 优点 | 风险 |
| --- | --- | --- |
| 固定长度 | 简单稳定 | 切断语义 |
| 按标题/段落 | 保留结构 | chunk 长度不均 |
| 滑窗重叠 | 减少边界丢失 | 重复内容增加 |
| 语义切分 | 更贴近自然段 | 实现复杂 |

推荐起点：

- 中文知识库：300-800 中文字符起步。
- 英文/代码：按 token 或结构切分。
- overlap：10%-20% 起步。
- 保留标题路径，例如 `产品手册 > 权限管理 > 角色配置`，写入 text 或 metadata。

面试时要能说：

> 如果 answer miss 很多，我会先看 chunk 是否太大或太小。太大时 embedding 表示被稀释，太小时答案上下文不完整。然后看 query 是否命中同义表达，必要时做 query rewrite、hybrid search 或 rerank。

## 9. 评估指标

离线检索指标：

- `recall@k`：正确 chunk 是否出现在 topK。
- `precision@k`：topK 中有多少相关。
- `MRR`：第一个正确结果排名是否靠前。
- `nDCG`：考虑多级相关性和排名质量。

RAG 端到端指标：

- answer hit rate：答案是否包含标准事实。
- groundedness：答案是否有证据支撑。
- faithfulness：是否幻觉。
- p50/p95/p99 latency。
- 每次查询 token 成本和 rerank 成本。

面试要点：

- 检索指标好不代表最终回答一定好，但检索指标差时最终回答很难好。
- 优先用标注 query-chunk 对做离线回归测试。
- 线上可以记录 query、召回 chunk、rerank 分数、最终答案和用户反馈。

## 10. 常见故障排查

### 召回结果明显不相关

检查顺序：

1. query 和 document 是否用同一个 embedding 模型。
2. embedding 维度是否一致。
3. metric 是否和模型要求一致。
4. 是否忘记归一化。
5. chunk 是否过长、过短或清洗错误。
6. filter 是否过严，导致候选太少。
7. ANN 参数是否过激，召回损失太大。

### 相关结果被权限外数据挤掉

解决：

- 在 Milvus search 中加权限 filter。
- 不要 search 后才在应用层过滤。
- 给常用权限字段建 scalar index。

### 延迟过高

排查：

- topK 是否过大。
- filter 字段是否没有索引或过滤选择性差。
- IVF 的 `nprobe` 或 HNSW 的 `ef` 是否过大。
- collection 是否 load 到内存。
- embedding 模型和 reranker 是否才是瓶颈。

### 更新后搜不到新内容

排查：

- 是否写入了正确 collection / partition。
- 是否 ID 冲突或 upsert 逻辑错误。
- 是否新版本被 filter 排除。
- 是否应用侧缓存没有刷新。
- 是否异步构建索引或加载状态未完成。

## 11. 面试高频问答

### Q1：Milvus 和普通数据库有什么区别？

普通数据库擅长精确条件查询和事务；Milvus 面向高维向量相似度检索，核心是 ANN 索引、向量距离计算、metadata filter 和大规模向量管理。在 RAG 中，普通数据库可以存文档和业务元数据，Milvus 负责从海量 chunk 中找语义相近内容。

### Q2：为什么向量检索需要索引？

高维向量逐条比对成本随数据量线性增长。索引用近似最近邻算法缩小搜索空间，把延迟降下来，但可能损失少量召回。工程上要用 recall@k 和 p95 latency 做权衡。

### Q3：IVF 和 HNSW 怎么选？

IVF 适合做大规模通用基线，参数直观，内存相对可控；HNSW 召回和延迟通常更好，但内存更高。小规模可用 FLAT 做真值评估；线上按数据量、更新频率、延迟预算和内存预算压测选择。

### Q4：RAG 为什么要 metadata filter？

因为语义相似不等于业务可用。企业知识库通常有租户、权限、时间、文档类型和版本约束。filter 能在检索阶段缩小候选，避免无权限或过期内容进入 topK。

### Q5：topK 取多大？

没有固定值。初始可以取 10-50 做召回，再 rerank 后取 3-8 个进 prompt。topK 太小容易漏召回，太大增加延迟和噪声。要用评估集扫 topK 曲线。

### Q6：embedding 模型升级要注意什么？

新旧 embedding 空间通常不可混用。需要重算文档向量，建立新 collection 或新 vector field，灰度切流并比较召回指标。query 和 document 必须使用同一套向量空间。

### Q7：为什么 dense + sparse hybrid 通常比单 dense 好？

dense 擅长语义相似，sparse/BM25 擅长关键词、编号、专有名词和代码符号。RAG 查询既有自然语言问题，也有精确术语，混合检索能提高覆盖面。

### Q8：如何判断是检索问题还是生成问题？

先看 topK chunk 是否包含答案。如果 topK 没有证据，是检索问题；如果证据在 topK 里但模型答错，是 prompt、rerank、上下文排序或生成模型问题。调试时保存 query、召回结果、分数和最终 prompt。

## 12. 实战练习

配套项目：[projects/07-milvus-rag](../projects/07-milvus-rag)。

运行后你要能解释：

1. collection 里有哪些字段，为什么这样设计。
2. `COSINE` metric 和向量归一化的关系。
3. 不加 filter 和加 `tenant_id` filter 的结果有什么不同。
4. topK 变化如何影响 recall。
5. 为什么 demo 用 hash embedding 只训练链路，真实项目要换成 embedding model。

完成标准：

- 能跑通 Milvus Lite 本地检索。
- 能改 chunk 文本和 metadata 后重新写入。
- 能构造一个 query，解释 top3 结果是否合理。
- 能口述 IVF、HNSW、filter、hybrid search、rerank 的取舍。
