.PHONY: schema ingest ingest-files ingest-tables rebuild-from-object-store walkthrough walkthrough-multimodal smoke api-smoke api-multimodal-smoke readiness-smoke security-smoke event-redaction-smoke auth-smoke context-smoke context-diagnosis-smoke context-backfill-smoke rewrite-smoke answer-eval-smoke answer-quality-smoke file-ingest-smoke table-ingest-smoke chunk-structure-smoke chunk-sweep-smoke search-param-smoke multimodal-search-smoke multimodal-prompt-smoke multimodal-eval-smoke multimodal-answer-smoke multimodal-answer-eval-smoke pdf-page-smoke heading-metadata-smoke feedback-eval-smoke eval-filter-smoke sparse-ablation-smoke retrieval-diagnosis-smoke lifecycle-smoke current-version-smoke current-version-unpublish-smoke embedding-model-smoke source-filter-smoke object-store-smoke object-store-delete-smoke observability-smoke monitoring-smoke benchmark-smoke release-gate-smoke container-config-smoke deploy-contract-smoke deploy-smoke eval answer-eval release-gate benchmark monitor export-feedback-eval sweep-chunking serve check llm-smoke milvus-smoke model-smoke pii-scan docs stats

schema:
	python schema.py --reset

ingest:
	python ingest_text.py
	python ingest_image.py

ingest-files:
	python ingest_files.py --input-dir ../../notes --tenant-id team_a --acl-group engineering

ingest-tables:
	python ingest_tables.py --input-dir ../../notes --tenant-id team_a --acl-group engineering

rebuild-from-object-store:
	python rebuild_from_object_store.py --reset

walkthrough:
	python walkthrough_core_rag.py

walkthrough-multimodal:
	python walkthrough_multimodal_rag.py

smoke:
	python smoke_e2e.py

api-smoke:
	python smoke_api.py

api-multimodal-smoke:
	python smoke_api_multimodal.py

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

context-diagnosis-smoke:
	python smoke_context_diagnosis.py

context-backfill-smoke:
	python smoke_context_backfill.py

rewrite-smoke:
	python smoke_rewrite.py

answer-eval-smoke:
	python smoke_answer_eval.py

answer-quality-smoke:
	python smoke_answer_quality_eval.py

file-ingest-smoke:
	python smoke_file_ingest.py

table-ingest-smoke:
	python smoke_table_ingest.py

chunk-structure-smoke:
	python smoke_chunk_structure.py

chunk-sweep-smoke:
	python smoke_chunk_sweep.py

search-param-smoke:
	python smoke_search_params.py

multimodal-search-smoke:
	python smoke_multimodal_search.py

multimodal-prompt-smoke:
	python smoke_multimodal_prompt.py

multimodal-eval-smoke:
	python smoke_multimodal_eval.py

multimodal-answer-smoke:
	python smoke_multimodal_answer.py

multimodal-answer-eval-smoke:
	python smoke_multimodal_answer_eval.py

pdf-page-smoke:
	python smoke_pdf_page_metadata.py

heading-metadata-smoke:
	python smoke_heading_metadata.py

feedback-eval-smoke:
	python smoke_feedback_eval_export.py

eval-filter-smoke:
	python smoke_eval_filters.py

sparse-ablation-smoke:
	python smoke_sparse_ablation.py

retrieval-diagnosis-smoke:
	python smoke_retrieval_diagnosis.py

lifecycle-smoke:
	python smoke_lifecycle.py

current-version-smoke:
	python smoke_current_version.py

current-version-unpublish-smoke:
	python smoke_current_version_unpublish.py

embedding-model-smoke:
	python smoke_embedding_model_filter.py

source-filter-smoke:
	python smoke_source_filter.py

object-store-smoke:
	python smoke_object_store_rebuild.py

object-store-delete-smoke:
	python smoke_object_store_delete_tombstone.py

observability-smoke:
	python smoke_observability.py

monitoring-smoke:
	python smoke_monitoring.py

benchmark-smoke:
	python smoke_benchmark.py

release-gate-smoke:
	python smoke_release_gate.py

container-config-smoke:
	python smoke_container_config.py

deploy-contract-smoke:
	python smoke_deploy_contract.py

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

export-feedback-eval:
	python build_eval_from_feedback.py --include-negative

sweep-chunking:
	python sweep_chunking.py --mode hybrid

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
