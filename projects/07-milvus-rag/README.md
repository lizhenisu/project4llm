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

## 面试口述版

RAG 中我会把文档切成 chunk，再把每个 chunk 的 embedding 和 metadata 写入 Milvus。查询时把用户问题编码为 query vector，结合租户、权限、文档来源等 metadata filter 做向量检索，得到候选 chunk。线上通常会把 topK 设得比最终上下文数量大，再接 reranker 或规则过滤，最后把高相关 chunk 送进 LLM。

## 注意

这个 demo 使用确定性 hash embedding，是为了不下载模型、稳定演示 Milvus 检索链路。真实 RAG 项目应替换为 `bge`、`e5`、`text-embedding` 等真实 embedding 模型，并用标注集评估 `recall@k`、MRR 和端到端回答质量。
