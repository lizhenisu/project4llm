# 09 Production RAG — Project Evaluation

> 本文档记录项目级测评结果，包括检索召回率实验、答案质量实验、压力测试等。运行产物、真实上传文件、评测集 JSONL、metrics JSON 和 details JSONL 不提交到仓库。

---

## 目录

1. [测评记录](#1-测评记录)
2. [2026-06-19 Retrieval Recall 测评](#2-2026-06-19-retrieval-recall-测评)
3. [后续压测记录](#3-后续压测记录)

---

## 1. 测评记录

| 日期 | 类型 | 项目版本 | 代码基线 | 结论 |
| --- | --- | --- | --- | --- |
| 2026-06-19 | Retrieval Recall | 0.3.3 | `dev` / `13caf86` | `hybrid` 模式 recall@10 = `0.9684` |

说明：

- 本轮 retrieval recall 测评结果归档到 `0.3.3`，用于覆盖上传解析可靠性、删除清理、评测脚本修正和项目测评文档等改进。

---

## 2. 2026-06-19 Retrieval Recall 测评

### 2.1 项目信息

| 项 | 值 |
| --- | --- |
| 项目 | `projects/09-production-rag` |
| 分支 | `dev` |
| 代码基线 | `13caf86` |
| 项目版本 | `0.3.3` |
| 评测脚本 | `eval_retrieval.py` |
| 评测模式 | `hybrid` |
| Top-K | `10` |

### 2.2 测试环境

| 项 | 值 |
| --- | --- |
| OS | WSL2 Linux `6.6.114.1-microsoft-standard-WSL2` |
| Python | `3.14.4` |
| Node.js | `v24.16.0` |
| npm | `11.13.0` |
| Milvus | `milvusdb/milvus:v2.6.13` |
| etcd | `quay.io/coreos/etcd:v3.5.18` |
| MinIO | `minio/minio:RELEASE.2024-05-28T17-19-04Z` |
| Milvus URI | `http://127.0.0.1:19530` |
| Collection | `rag_chunks_v1` |

### 2.3 测试数据

| 项 | 值 |
| --- | --- |
| 用户/租户 | 固定测试租户 `tenant-fixed-test` |
| 文档数量 | `20` 个 PDF |
| Query 数量 | `380` |
| 每个 PDF 的 Query 数量 | `19` |
| 评测粒度 | 文档级召回 |
| 期望目标 | 每条 query 对应 1 个 `expected_doc_id` |

评测集由当前测试环境中已上传并解析完成的 PDF 生成。评测集文件位于运行时目录 `runtime/retrieval_recall_eval.jsonl`，属于本地运行产物，不提交仓库。

### 2.4 测试方法

执行的完整评测命令：

```bash
cd projects/09-production-rag
source ../../.venv/bin/activate

RAG_MILVUS_URI=http://127.0.0.1:19530 \
RAG_OBJECT_STORE_DIR=$PWD/object_store \
RAG_RUNTIME_DIR=$PWD/runtime \
PYTHONPATH=. \
python -u eval_retrieval.py \
  --mode hybrid \
  --limit 10 \
  --include-all-sources \
  --require-real-api \
  --input runtime/retrieval_recall_eval.jsonl \
  --json-output runtime/retrieval_recall_metrics_hybrid.json \
  --details-output runtime/retrieval_recall_details_hybrid.jsonl
```

检索链路：

```text
query -> real embedding API -> Milvus hybrid retrieval -> top-K results
```

本次完成的 `hybrid` 模式使用真实 SiliconFlow embedding API 和 Milvus 混合检索，不调用最终回答 LLM。

完整生产 `rerank` 链路也尝试运行过：

```text
query rewrite -> embedding -> Milvus hybrid retrieval -> rerank -> top-K results
```

但 `rerank` 批量评测在中途被上游 LLM API 限流中断：

```text
openai.RateLimitError: Error code: 429 - Resource exhausted
```

因此本次可完整复现和落盘的正式结果采用 `hybrid` 模式。

### 2.5 指标定义

| 指标 | 计算方式 | 含义 |
| --- | --- | --- |
| `recall@10` | top 10 中至少命中 1 个期望文档的 query 数 / 总 query 数 | 文档是否被召回 |
| `macro_target_recall@10` | 每条 query 的目标召回率平均值 | 单 query 维度的平均召回 |
| `micro_target_recall@10` | 总命中目标数 / 总期望目标数 | 全局目标召回 |
| `MRR@10` | 第一个命中结果排名倒数的平均值 | 正确文档是否排得靠前 |
| `nDCG@10` | DCG / IDCG | 排序质量 |
| `avg_latency_ms` | 总延迟平均值 | 平均检索耗时 |
| `p95_latency_ms` | 95 分位延迟 | 尾部延迟 |
| `stage_p95_latency_ms` | 各阶段 95 分位延迟 | 瓶颈定位 |
| `permission_leakage_failures` | 返回非当前租户结果的失败次数 | 权限隔离检查 |

本轮评测还修正了两个评估脚本问题：

- Milvus 返回页级 `doc_id` 时，评测脚本会归一到源文档 ID 再计算文档级 recall。
- `nDCG@10` 对同一 expected 目标只计一次，避免同一文档多个页命中导致 nDCG 大于 1。

### 2.6 测试结果

| 指标 | 结果 |
| --- | ---: |
| `mode` | `hybrid` |
| `limit` | `10` |
| `query_count` | `380` |
| `answerable_count` | `380` |
| `recall@10` | `0.9684` |
| `macro_target_recall@10` | `0.9684` |
| `micro_target_recall@10` | `0.9684` |
| `micro_expected_total` | `380` |
| `micro_matched_total` | `368` |
| `MRR@10` | `0.9007` |
| `nDCG@10` | `0.9175` |
| `avg_latency_ms` | `261.29` |
| `p95_latency_ms` | `258.36` |
| `embedding p95 ms` | `252.16` |
| `milvus_search p95 ms` | `7.06` |
| `permission_leakage_failures` | `0` |

### 2.7 结果解读

- `368 / 380` 条 query 在 top 10 中召回了期望文档，整体 recall@10 为 `96.84%`。
- MRR@10 为 `0.9007`，说明大多数命中文档排在较靠前的位置。
- Milvus 检索 p95 仅 `7.06ms`，主要耗时来自外部 embedding API，embedding p95 为 `252.16ms`。
- 权限泄漏失败为 `0`，本轮未发现跨租户返回问题。
- 未命中的 12 条 query 多数是宽泛概念问题，例如 `long-term forecasting`、`time series forecasting`、`channel independence`、`attention across variables`。这类 query 本身可对应多篇论文，不一定是纯检索失败，也提示后续评测集应区分“唯一定位型 query”和“概念竞争型 query”。

### 2.8 本地运行产物

以下文件为本地运行产物，已被 `.gitignore` 排除：

- `runtime/retrieval_recall_eval.jsonl`
- `runtime/retrieval_recall_metrics_hybrid.json`
- `runtime/retrieval_recall_details_hybrid.jsonl`
- `runtime/retrieval_recall_report_hybrid.md`

---

## 3. 后续压测记录

后续压力测试结果也记录在本文档中。建议压测记录至少包含：

- 项目版本和 commit。
- 部署方式和机器规格。
- 测试数据规模。
- 并发策略。
- 成功率、错误率、吞吐量、平均延迟、p95/p99 延迟。
- CPU、内存、磁盘、网络等资源瓶颈。
- 是否使用真实外部模型 API，或是否使用 mock LLM / mock embedding。
