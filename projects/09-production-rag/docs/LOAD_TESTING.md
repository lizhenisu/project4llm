# Production RAG Pressure Test

Run one command from the repository root:

```bash
projects/09-production-rag/scripts/deploy_and_load_test_2c2g.sh \
  --min-concurrency 1 \
  --max-concurrency 64 \
  --include-source-identifiers
```

This command deploys the project in production-container mode, replaces external LLM / embedding / rerank calls with local mock services, tests the machine's RAG serving capacity, prints the final limit in the terminal, writes a report, and then stops the test containers.

The only script options are:

- `--min-concurrency N`: lowest concurrency to start from. Default: `1`.
- `--max-concurrency N`: highest concurrency to search up to. Default: `64`.
- `--include-source-identifiers`: include limited `doc_id` / `doc_version` details in the report for debugging. Do not commit reports containing identifiers.

## Method

The test only measures the RAG query path. Before pressure testing, the script checks `/sources` with the test token and aborts if no ready document can be retrieved.

By default the load test uses all visible ready documents for the test user, not only documents marked current. If the user can see 20 ready PDFs, the RAG request is allowed to retrieve from all 20.

The test uses adaptive concurrency search:

1. Probe upward by doubling concurrency: `1, 2, 4, 8...`.
2. Stop when the first unusable level appears or `--max-concurrency` is reached.
3. Binary-search between the last usable level and the first unusable level.
4. Print the approximate highest usable concurrency.

Each concurrency level sends:

```text
requests = max(20, concurrency * 5)
```

There is no default request cap. This keeps the target concurrency meaningful. For example, concurrency `10000` sends `50000` requests.

## Pass Criteria

A concurrency level is usable only when all checks pass:

```text
failure_rate <= 1%
p95_latency_ms <= 30000
first_event_p95_ms <= 5000
citations_avg > 0
```

The citation check is required so the test cannot accidentally measure direct chat or an empty knowledge base and call it RAG pressure testing.

## Metrics

The final terminal output reports:

- `Highest usable concurrency`: highest concurrency level that passed all checks.
- `First unusable concurrency`: first concurrency level that failed, if found.
- `P95 latency`: 95% of requests completed within this latency.
- `Throughput`: completed requests per second.
- `Citations avg`: average number of citations returned per request.
- `Report`: Markdown report path.
- `Summary JSON`: machine-readable result path.

Metric formulas:

```text
success_rate = success / requests
failure_rate = failed / requests
throughput_rps = success / wall_time_seconds
latency_ms = request_end_time - request_start_time
p95_latency_ms = 95th percentile of latency_ms
first_event_ms = first_stream_event_time - request_start_time
first_event_p95_ms = 95th percentile of first_event_ms
citations_avg = total_citations / successful_requests
```

## Output

Results are written to:

```text
projects/09-production-rag/docs/load-tests/production-limit-YYYYMMDD-HHMMSS/
```

The directory contains:

- `report.md`: human-readable report.
- `summary.json`: machine-readable summary.
- `concurrency-*.json`: result for each tested concurrency level.
- `concurrency-*.stdout` and `concurrency-*.stderr`: raw runner logs for each level.

The report directory is runtime output. Do not commit it if it contains source identifiers, filenames, or other user-private data.

## Conversation API load

To isolate PostgreSQL conversation persistence from model and Milvus latency, run
the create/update/list/read/delete workflow against one or more API instances:

```bash
source .venv/bin/activate
python projects/09-production-rag/tests/load/conversation_api_load.py \
  --base-urls http://127.0.0.1:18181,http://127.0.0.1:18182 \
  --token synthetic-test-token \
  --users 1000 \
  --concurrency 100
```

Each virtual user owns a synthetic tenant and a unique conversation. Consecutive
operations rotate across the supplied API origins, validate read-after-write
consistency, and delete the conversation at the end. The summary reports workflow
and request throughput plus per-operation p50/p95/p99 latency. Use a synthetic
token and tenant prefix, and do not commit output from runs against real users.
