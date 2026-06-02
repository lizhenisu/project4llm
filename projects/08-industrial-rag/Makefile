.PHONY: schema ingest ingest-files rebuild-from-object-store smoke api-smoke readiness-smoke security-smoke event-redaction-smoke auth-smoke context-smoke rewrite-smoke answer-eval-smoke file-ingest-smoke lifecycle-smoke current-version-smoke embedding-model-smoke source-filter-smoke object-store-smoke observability-smoke monitoring-smoke release-gate-smoke deploy-smoke eval answer-eval release-gate benchmark monitor serve check llm-smoke milvus-smoke model-smoke pii-scan docs stats

schema:
	python schema.py --reset

ingest:
	python ingest_text.py
	python ingest_image.py

ingest-files:
	python ingest_files.py --input-dir ../../notes --tenant-id team_a --acl-group engineering

rebuild-from-object-store:
	python rebuild_from_object_store.py --reset

smoke:
	python smoke_e2e.py

api-smoke:
	python smoke_api.py

readiness-smoke:
	python smoke_readiness.py

security-smoke:
	python smoke_security.py

event-redaction-smoke:
	python smoke_event_redaction.py

auth-smoke:
	python smoke_auth_context.py

context-smoke:
	python smoke_context.py

rewrite-smoke:
	python smoke_rewrite.py

answer-eval-smoke:
	python smoke_answer_eval.py

file-ingest-smoke:
	python smoke_file_ingest.py

lifecycle-smoke:
	python smoke_lifecycle.py

current-version-smoke:
	python smoke_current_version.py

embedding-model-smoke:
	python smoke_embedding_model_filter.py

source-filter-smoke:
	python smoke_source_filter.py

object-store-smoke:
	python smoke_object_store_rebuild.py

observability-smoke:
	python smoke_observability.py

monitoring-smoke:
	python smoke_monitoring.py

release-gate-smoke:
	python smoke_release_gate.py

pii-scan:
	python scan_pii.py data/sample_docs.jsonl data/sample_images.jsonl --fail

deploy-smoke:
	python smoke_deploy.py

eval:
	python eval_retrieval.py

answer-eval:
	python eval_answer.py

release-gate:
	python release_gate.py

benchmark:
	python benchmark_latency.py

monitor:
	python monitor_events.py

serve:
	uvicorn serve:app --host 127.0.0.1 --port 8008

check:
	python check_config.py

docs:
	python list_documents.py --tenant-id team_a

stats:
	python collection_stats.py --tenant-id team_a

llm-smoke:
	python smoke_llm.py

milvus-smoke:
	python smoke_milvus.py

model-smoke:
	python smoke_models.py
