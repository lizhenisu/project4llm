from __future__ import annotations

import re
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
ALERTS_PATH = PROJECT_DIR / "ops" / "prometheus" / "alerts.yml"
EXPECTED_ALERTS = {
    "ProductionRagApiDown",
    "ProductionRagHighHttp5xxRate",
    "ProductionRagHttpP95LatencyHigh",
    "ProductionRagQueryStreamRejections",
    "ProductionRagQueryRateLimitRejections",
    "ProductionRagQueryRecoveryStaleProcessing",
    "ProductionRagModelApiFailureRateHigh",
    "ProductionRagModelUsageRecordingFailure",
    "ProductionRagMetadataPoolTimeout",
    "ProductionRagIngestionBacklogHigh",
}
EXPORTED_METRICS = {
    "up",
    "rag_http_requests_total",
    "rag_http_responses_total",
    "rag_http_request_latency_seconds_bucket",
    "rag_query_stream_events_total",
    "rag_query_rate_limit_events_total",
    "rag_query_result_stale_processing_entries",
    "rag_model_api_operation_calls_total",
    "rag_model_usage_recording_events_total",
    "rag_metadata_pool_timeouts_total",
    "rag_ingestion_tasks",
}


def main() -> None:
    payload = yaml.safe_load(ALERTS_PATH.read_text(encoding="utf-8"))
    groups = payload.get("groups") or []
    assert len(groups) == 1
    rules = groups[0].get("rules") or []
    assert {rule["alert"] for rule in rules} == EXPECTED_ALERTS

    for rule in rules:
        assert rule.get("for")
        assert rule.get("labels", {}).get("severity") in {"warning", "critical"}
        assert rule.get("annotations", {}).get("summary")
        expression = str(rule.get("expr") or "")
        assert expression.strip()
        referenced = {
            token
            for token in re.findall(r"\b[a-zA-Z_:][a-zA-Z0-9_:]*\b", expression)
            if token.startswith("rag_") or token == "up"
        }
        assert referenced <= EXPORTED_METRICS, (rule["alert"], referenced - EXPORTED_METRICS)
        assert "tenant_id" not in expression
        assert "user_id" not in expression

    rules_by_name = {rule["alert"]: rule for rule in rules}
    assert (
        str(rules_by_name["ProductionRagQueryRecoveryStaleProcessing"]["expr"]).strip()
        == "max(rag_query_result_stale_processing_entries) > 0"
    )
    assert (
        str(rules_by_name["ProductionRagIngestionBacklogHigh"]["expr"]).strip()
        == 'sum(max by (status) (rag_ingestion_tasks{status=~"queued|processing"})) > 1000'
    )
    print("smoke_prometheus_alerts=ok")


if __name__ == "__main__":
    main()
