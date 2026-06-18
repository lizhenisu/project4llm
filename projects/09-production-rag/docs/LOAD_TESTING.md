# Production RAG Load Testing

This project has two load-test modes:

- **mock external API mode**: test the RAG service, Milvus, streaming events, and UI-facing behavior with controllable LLM / embedding / rerank latency.
- **real external API mode**: run a small number of end-to-end requests against the real model providers to validate timeout, rate-limit, and user-visible behavior.

Use mock mode for normal pressure testing. Use real mode only with low concurrency because external model APIs add cost, provider-side rate limits, and network noise.

## 1. Start The Mock External API

From the repository root:

```bash
source .venv/bin/activate
cd projects/09-production-rag
uvicorn tests.load.mock_external_api:app --host 127.0.0.1 --port 18080
```

Useful knobs:

```bash
MOCK_LLM_LATENCY_MS=1200
MOCK_EMBEDDING_LATENCY_MS=200
MOCK_RERANK_LATENCY_MS=500
MOCK_LLM_ERROR_RATE=0.01
MOCK_EMBEDDING_ERROR_RATE=0.01
MOCK_RERANK_ERROR_RATE=0.01
MOCK_EMBEDDING_DIM=1024
```

## 2. Start The Backend Against The Mock API

Use the normal development backend, but point external calls to the mock service:

```bash
source ../../.venv/bin/activate
NEW_API_URL=http://127.0.0.1:18080/v1 \
NEW_API_KEY=mock-key \
SILICONFLOW_URL=http://127.0.0.1:18080 \
SILICONFLOW_API_KEY=mock-key \
uvicorn serve:app --host 0.0.0.0 --port 8008
```

For stable load-test measurements, avoid `--reload`.

## 3. Run Query Load

Direct LLM mode, without retrieval:

```bash
python tests/load/rag_query_load.py \
  --base-url http://127.0.0.1:8008 \
  --token production-rag-fixed-test-login-token \
  --external-mode mock \
  --concurrency 5 \
  --requests 50 \
  --question "总结这些资料的核心内容"
```

RAG retrieval mode requires selecting documents by doc id, source type, or doc version. Without a document filter, the backend intentionally uses direct LLM chat mode.

```bash
python tests/load/rag_query_load.py \
  --base-url http://127.0.0.1:8008 \
  --token production-rag-fixed-test-login-token \
  --external-mode mock \
  --concurrency 5 \
  --requests 50 \
  --source-type pdf \
  --question "总结这些资料的核心内容"
```

The summary includes:

- `latency_ms`: end-to-end `/query/stream` latency.
- `first_event_ms`: time until the first NDJSON event arrives.
- `stage_latency_ms`: backend-reported stage latency.
- `external_latency_ms.llm`: query rewrite plus final answer generation.
- `external_latency_ms.embedding`: text/image embedding stage latency.
- `external_latency_ms.rerank`: rerank stage latency.
- `unattributed_ms`: total latency minus reported stage latency, useful for queueing, HTTP overhead, and missing instrumentation.

## 4. Real External API Sanity Check

Keep this small:

```bash
python tests/load/rag_query_load.py \
  --base-url http://127.0.0.1:8008 \
  --token production-rag-fixed-test-login-token \
  --external-mode real \
  --concurrency 1 \
  --requests 5 \
  --question "总结这些资料的核心内容"
```

Then try concurrency `2` and `3` if provider limits and cost allow it. Do not use the real provider for high-concurrency pressure tests.

## 5. Deployed 2C2G Limit Test

When the project is already deployed with containers on a real 2-core / 2GB server, run the limit test script from the repository root:

```bash
source .venv/bin/activate
export RAG_LOAD_TEST_TOKEN="production-rag-fixed-test-login-token"

projects/09-production-rag/scripts/run_2c2g_load_test.sh \
  --base-url http://127.0.0.1:8008 \
  --question "总结这些资料的核心内容"
```

The script ramps through concurrency levels instead of doing a tiny smoke check. Defaults:

```text
1, 2, 4, 8, 16, 32, 64
```

For each level it runs:

```text
max(20, concurrency * 5) requests, capped at 320 requests
```

A level is considered usable only when:

- failure rate is at most `1%`
- P95 request latency is at most `30000 ms`
- P95 first event latency is at most `5000 ms`
- average citation count is greater than `0`

The citation requirement is intentional: it avoids accidentally measuring direct chat mode or an empty knowledge base while calling it a RAG pressure test.

Before ramping concurrency, the script calls `/sources` with the same token and records the visible data scale:

- total visible sources and chunks
- visible sources by `source_type`, such as `pdf`, `txt`, `md`
- visible sources by status, such as `ready`, `processing`, `failed`
- how many sources/chunks match the current `source_type`, `doc_id`, and `doc_version` filters

Only counts are written to the report. Source titles and filenames are intentionally omitted.

To push harder:

```bash
projects/09-production-rag/scripts/run_2c2g_load_test.sh \
  --base-url http://127.0.0.1:8008 \
  --concurrency-levels 1,2,4,8,16,32,64,96,128 \
  --requests-per-concurrency 8 \
  --max-requests 512 \
  --max-p95-ms 45000
```

To target specific documents:

```bash
projects/09-production-rag/scripts/run_2c2g_load_test.sh \
  --base-url http://127.0.0.1:8008 \
  --doc-id "your-doc-id"
```

By default, the script uses common text/document source types:

```text
pdf, txt, md, html, table, csv, tsv
```

Raw JSON, stdout/stderr per stage, and the Markdown report are written to:

```text
projects/09-production-rag/docs/load-tests/production-limit-YYYYMMDD-HHMMSS/
```

Use `--dry-run` to inspect the exact commands before starting the test.
