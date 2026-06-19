# Retrieval Recall Experiment

This experiment measures whether retrieval returns the expected documents or chunks. It does not build the final answer prompt and does not call the answer-generation LLM.

Use real provider APIs for the retrieval path:

```bash
cd projects/09-production-rag
source ../../.venv/bin/activate

python eval_retrieval.py \
  --mode rerank \
  --limit 10 \
  --include-all-sources \
  --require-real-api \
  --input runtime/retrieval_recall_eval.jsonl \
  --json-output runtime/retrieval_recall_metrics.json \
  --details-output runtime/retrieval_recall_details.jsonl
```

`--mode rerank` runs the production retrieval chain:

```text
query rewrite -> embedding -> Milvus hybrid retrieval -> rerank -> top-K hits
```

It stops before context packing and final answer generation.

## Eval Set

Create `runtime/retrieval_recall_eval.jsonl` manually. Each line is one query:

```jsonl
{"query":"这篇自然辩证法资料主要讨论什么？","tenant_id":"team_a","source_types":["pdf"],"expected_doc_ids":["your-doc-id"],"answerable":true}
{"query":"某个章节里的具体概念是什么？","tenant_id":"team_a","source_types":["pdf"],"expected_chunk_ids":["your-doc-id:12"],"answerable":true}
```

Use `expected_doc_ids` for document-level recall. Use `expected_chunk_ids` for stricter chunk-level recall.

Do not commit eval sets if they contain real filenames, private document names, or user content.

## Metrics

- `recall@K`: query-level hit rate. A query counts as hit when top-K contains at least one expected document/chunk.
- `macro_target_recall@K`: average per-query target recall.
- `micro_target_recall@K`: total matched expected targets divided by total expected targets.
- `mrr@K`: reciprocal rank of the first matched target.
- `ndcg@K`: ranking quality when there can be multiple expected targets.
- `stage_p95_latency_ms`: p95 latency by retrieval stage.

For a real recall experiment, prefer `macro_target_recall@K` and `micro_target_recall@K` over the older `recall@K` field because they measure how many expected targets were recovered, not just whether each query hit at least one target.

## Scope

`--include-all-sources` means retrieval can search every active source visible to the tenant and ACL context instead of only current documents. Omit it only when the experiment intentionally measures the current-document subset.

`--require-real-api` fails fast unless the configured retrieval path uses real external APIs:

- `RAG_EMBEDDING_BACKEND=siliconflow`
- `RAG_RERANK_BACKEND=siliconflow` for `--mode rerank`
- `RAG_QUERY_REWRITE_BACKEND=llm`
- `SILICONFLOW_API_KEY`, `NEW_API_URL`, and `NEW_API_KEY` are configured

