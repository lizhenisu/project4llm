.PHONY: schema serve check ingest-files ingest-tables rebuild stats monitor eval answer-eval release-gate benchmark smoke api-smoke deploy-smoke frontend-install frontend-dev frontend-build compose-up compose-down

schema:
	python schema.py --reset

serve:
	uvicorn serve:app --host 0.0.0.0 --port $${RAG_API_PORT:-8008}

check:
	python check_config.py

ingest-files:
	python ingest_files.py --input-dir "$${RAG_INPUT_DIR}" --tenant-id "$${RAG_TENANT_ID}" --acl-group "$${RAG_ACL_GROUP:-engineering}"

ingest-tables:
	python ingest_tables.py --input-dir "$${RAG_TABLE_DIR}" --tenant-id "$${RAG_TENANT_ID}" --acl-group "$${RAG_ACL_GROUP:-engineering}"

rebuild:
	python rebuild_from_object_store.py --reset

stats:
	python collection_stats.py

monitor:
	python monitor_events.py

eval:
	python eval_retrieval.py

answer-eval:
	python eval_answer.py

release-gate:
	python release_gate.py

benchmark:
	python benchmark_latency.py

smoke:
	python tests/smoke/smoke_e2e.py

api-smoke:
	python tests/smoke/smoke_api.py

deploy-smoke:
	python tests/smoke/smoke_deploy.py

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev -- --host 0.0.0.0

frontend-build:
	cd frontend && npm run build

compose-up:
	docker compose up -d milvus rag-api rag-web

compose-down:
	docker compose down
