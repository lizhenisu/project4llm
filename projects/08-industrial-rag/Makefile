.PHONY: schema ingest smoke api-smoke security-smoke context-smoke rewrite-smoke answer-eval-smoke deploy-smoke eval answer-eval benchmark serve check llm-smoke milvus-smoke model-smoke pii-scan

schema:
	python schema.py --reset

ingest:
	python ingest_text.py
	python ingest_image.py

smoke:
	python smoke_e2e.py

api-smoke:
	python smoke_api.py

security-smoke:
	python smoke_security.py

context-smoke:
	python smoke_context.py

rewrite-smoke:
	python smoke_rewrite.py

answer-eval-smoke:
	python smoke_answer_eval.py

pii-scan:
	python scan_pii.py data/sample_docs.jsonl data/sample_images.jsonl --fail

deploy-smoke:
	python smoke_deploy.py

eval:
	python eval_retrieval.py

answer-eval:
	python eval_answer.py

benchmark:
	python benchmark_latency.py

serve:
	uvicorn serve:app --host 127.0.0.1 --port 8008

check:
	python check_config.py

llm-smoke:
	python smoke_llm.py

milvus-smoke:
	python smoke_milvus.py

model-smoke:
	python smoke_models.py
