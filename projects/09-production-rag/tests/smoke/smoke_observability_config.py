from __future__ import annotations

import json
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
COMPOSE_PATH = PROJECT_DIR / "docker-compose.yml"
PROMETHEUS_PATH = PROJECT_DIR / "ops" / "prometheus" / "prometheus.yml"
PROMETHEUS_TESTS_PATH = PROJECT_DIR / "ops" / "prometheus" / "tests.yml"
GRAFANA_DATASOURCE_PATH = (
    PROJECT_DIR / "ops" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
)
GRAFANA_DASHBOARD_PROVIDER_PATH = (
    PROJECT_DIR / "ops" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
)
GRAFANA_DASHBOARD_PATH = PROJECT_DIR / "ops" / "grafana" / "dashboards" / "production-rag.json"


def main() -> None:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    prometheus = yaml.safe_load(PROMETHEUS_PATH.read_text(encoding="utf-8"))
    prometheus_tests = yaml.safe_load(PROMETHEUS_TESTS_PATH.read_text(encoding="utf-8"))
    datasource = yaml.safe_load(GRAFANA_DATASOURCE_PATH.read_text(encoding="utf-8"))
    dashboard_provider = yaml.safe_load(GRAFANA_DASHBOARD_PROVIDER_PATH.read_text(encoding="utf-8"))
    dashboard = json.loads(GRAFANA_DASHBOARD_PATH.read_text(encoding="utf-8"))

    service = compose["services"]["prometheus"]
    assert service["image"] == "prom/prometheus:v3.11.3"
    assert service["profiles"] == ["observability"]
    assert service["depends_on"]["rag-api"]["condition"] == "service_healthy"
    assert "prometheus-data:/prometheus" in service["volumes"]
    assert "./ops/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" in service["volumes"]
    assert "./ops/prometheus/alerts.yml:/etc/prometheus/alerts.yml:ro" in service["volumes"]
    assert "prometheus-data" in compose["volumes"]

    grafana = compose["services"]["grafana"]
    assert grafana["image"] == "grafana/grafana:13.0.3"
    assert grafana["profiles"] == ["observability"]
    assert "grafana-data:/var/lib/grafana" in grafana["volumes"]
    assert "./ops/grafana/provisioning:/etc/grafana/provisioning:ro" in grafana["volumes"]
    assert "./ops/grafana/dashboards:/var/lib/grafana/dashboards:ro" in grafana["volumes"]
    assert "grafana-data" in compose["volumes"]

    assert prometheus["global"]["scrape_interval"] == "15s"
    assert prometheus["global"]["evaluation_interval"] == "30s"
    assert prometheus["rule_files"] == ["/etc/prometheus/alerts.yml"]
    api_job = next(job for job in prometheus["scrape_configs"] if job["job_name"] == "production-rag-api")
    assert api_job["metrics_path"] == "/metrics"
    assert "static_configs" not in api_job
    assert api_job["dns_sd_configs"] == [
        {
            "names": ["rag-api"],
            "type": "A",
            "port": 8008,
            "refresh_interval": "15s",
        }
    ]
    dedup_test = prometheus_tests["tests"][0]
    dedup_expressions = {
        case["expr"]: case["exp_samples"][0]["value"]
        for case in dedup_test["promql_expr_test"]
    }
    assert dedup_expressions == {
        'sum(max by (status) (rag_ingestion_tasks{status=~"queued|processing"}))': 900,
        "max(rag_query_result_stale_processing_entries)": 1,
        "max by (scope) (rag_query_rate_limit_requests)": 42,
    }

    configured_source = datasource["datasources"][0]
    assert configured_source["uid"] == "prometheus"
    assert configured_source["url"] == "http://prometheus:9090"
    assert configured_source["isDefault"] is True
    assert dashboard_provider["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"

    assert dashboard["uid"] == "production-rag-concurrency"
    assert dashboard["title"] == "Production RAG Concurrent Workloads"
    assert dashboard["refresh"] == "30s"
    assert len(dashboard["panels"]) >= 10
    expressions = [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    ]
    assert any("histogram_quantile" in expression for expression in expressions)
    assert any("rag_query_stream_events_total" in expression for expression in expressions)
    assert any("rag_query_rate_limit_events_total" in expression for expression in expressions)
    assert any("rag_model_api_operation_calls_total" in expression for expression in expressions)
    assert any("rag_ingestion_tasks" in expression for expression in expressions)
    assert any("rag_ingestion_operator_audit_events" in expression for expression in expressions)
    assert any("rag_ingestion_stage_duration_seconds_average" in expression for expression in expressions)
    assert any("rag_ingestion_stage_samples" in expression for expression in expressions)
    assert (
        'sum(max by (status) (rag_ingestion_tasks{status=~"queued|processing"}))'
        in expressions
    )
    assert "max by (status) (rag_ingestion_tasks)" in expressions
    pressure_panel = next(
        panel for panel in dashboard["panels"]
        if panel["title"] == "Ingestion and Metadata Pressure"
    )
    assert [target["expr"] for target in pressure_panel["targets"]] == [
        "max by (status) (rag_ingestion_tasks)",
        "increase(rag_metadata_pool_timeouts_total[$__rate_interval])",
        'max by (outcome) (rag_ingestion_operator_audit_events{operation="bulk_redrive"})',
    ]
    image_size_panel = next(
        panel for panel in dashboard["panels"]
        if panel["title"] == "Query Image Payload Size p95"
    )
    assert image_size_panel["fieldConfig"]["defaults"]["unit"] == "bytes"
    assert "rag_query_image_payload_bytes_bucket" in image_size_panel["targets"][0]["expr"]
    assert "histogram_quantile(0.95" in image_size_panel["targets"][0]["expr"]
    eta_panel = next(
        panel for panel in dashboard["panels"]
        if panel["title"] == "Ingestion ETA Stage History"
    )
    assert eta_panel["fieldConfig"]["defaults"]["unit"] == "s"
    eta_expressions = [target["expr"] for target in eta_panel["targets"]]
    assert eta_expressions == [
        "max by (source_type, stage) (rag_ingestion_stage_duration_seconds_average)",
        "max by (source_type, stage) (rag_ingestion_stage_samples)",
    ]
    recovery_panel = next(
        panel for panel in dashboard["panels"]
        if panel["title"] == "Query Recovery Cache"
    )
    recovery_expressions = [target["expr"] for target in recovery_panel["targets"]]
    assert recovery_expressions == [
        "max by (status) (rag_query_result_cache_entries)",
        "max(rag_query_result_cache_expired_entries)",
        "max(rag_query_result_stale_processing_entries)",
        "max(rag_query_result_events)",
    ]
    rate_limit_panel = next(
        panel for panel in dashboard["panels"]
        if panel["title"] == "Query Request Rate Limits"
    )
    rate_limit_expressions = [target["expr"] for target in rate_limit_panel["targets"]]
    assert rate_limit_expressions == [
        "max by (scope) (rag_query_rate_limit_config)",
        "max by (scope) (rag_query_rate_limit_requests)",
        "max by (scope) (rag_query_rate_limit_active_keys)",
        "sum by (event) (rate(rag_query_rate_limit_events_total[$__rate_interval]))",
    ]
    print("smoke_observability_config=ok")


if __name__ == "__main__":
    main()
