from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.model_api_retry import (
    call_model_api_with_retries,
    chat_completion_with_fallback,
    model_api_metrics_snapshot,
    reset_llm_failover_state,
)


class ApiError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"api status {status_code}")
        self.status_code = status_code


def main() -> None:
    old_retries = os.environ.get("RAG_MODEL_API_RETRIES")
    old_backoff = os.environ.get("RAG_MODEL_API_BACKOFF_SECONDS")
    old_max_backoff = os.environ.get("RAG_MODEL_API_BACKOFF_MAX_SECONDS")
    try:
        os.environ["RAG_MODEL_API_RETRIES"] = "3"
        os.environ["RAG_MODEL_API_BACKOFF_SECONDS"] = "0"
        os.environ["RAG_MODEL_API_BACKOFF_MAX_SECONDS"] = "0"
        test_transient_errors_retry_until_success()
        test_non_transient_errors_do_not_retry()
        test_transient_primary_failure_uses_fallback_and_circuit()
    finally:
        restore_env("RAG_MODEL_API_RETRIES", old_retries)
        restore_env("RAG_MODEL_API_BACKOFF_SECONDS", old_backoff)
        restore_env("RAG_MODEL_API_BACKOFF_MAX_SECONDS", old_max_backoff)
    print("smoke_model_api_retry=ok")


def test_transient_errors_retry_until_success() -> None:
    attempts = {"count": 0}
    before = operation_metrics("flaky")

    def flaky_call() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ApiError(429)
        return "ok"

    with patch("rag_core.model_api_retry.time.sleep", return_value=None):
        assert call_model_api_with_retries("flaky", flaky_call) == "ok"
    assert attempts["count"] == 3
    after = operation_metrics("flaky")
    assert after["calls_total"] == before["calls_total"] + 1
    assert after["attempts_total"] == before["attempts_total"] + 3
    assert after["retries_total"] == before["retries_total"] + 2
    assert after["successes_total"] == before["successes_total"] + 1
    assert after["failures_total"] == before["failures_total"]


def test_non_transient_errors_do_not_retry() -> None:
    attempts = {"count": 0}
    before = operation_metrics("bad_request")

    def bad_request() -> str:
        attempts["count"] += 1
        raise ApiError(400)

    try:
        call_model_api_with_retries("bad_request", bad_request)
    except ApiError as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("expected ApiError")
    assert attempts["count"] == 1
    after = operation_metrics("bad_request")
    assert after["calls_total"] == before["calls_total"] + 1
    assert after["attempts_total"] == before["attempts_total"] + 1
    assert after["retries_total"] == before["retries_total"]
    assert after["failures_total"] == before["failures_total"] + 1
    assert after["latency_max_ms"] >= 0


def test_transient_primary_failure_uses_fallback_and_circuit() -> None:
    calls: list[tuple[str, str]] = []

    class FakeOpenAI:
        def __init__(self, *, base_url: str, api_key: str, **_kwargs) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, *, model: str, messages: list[dict], **kwargs):
            calls.append((self.base_url, model))
            if self.base_url == "https://primary.example/v1":
                raise ApiError(502)
            assert self.base_url == "https://fallback.example/v1"
            assert self.api_key == "fallback-key"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    config = SimpleNamespace(
        llm_base_url="https://primary.example/v1",
        llm_api_key="primary-key",
        llm_model="primary-model",
        siliconflow_base_url="https://api.siliconflow.cn",
        siliconflow_api_key=None,
    )
    old_values = {
        name: os.environ.get(name)
        for name in (
            "RAG_MODEL_API_RETRIES",
            "RAG_LLM_FALLBACK_BASE_URL",
            "RAG_LLM_FALLBACK_API_KEY",
            "RAG_LLM_FALLBACK_MODEL",
            "RAG_LLM_FALLBACK_BACKEND",
            "RAG_LLM_FAILOVER_COOLDOWN_SECONDS",
        )
    }
    try:
        os.environ["RAG_MODEL_API_RETRIES"] = "1"
        os.environ["RAG_LLM_FALLBACK_BASE_URL"] = "https://fallback.example"
        os.environ["RAG_LLM_FALLBACK_API_KEY"] = "fallback-key"
        os.environ["RAG_LLM_FALLBACK_MODEL"] = "fallback-model"
        os.environ["RAG_LLM_FALLBACK_BACKEND"] = "backup"
        os.environ["RAG_LLM_FAILOVER_COOLDOWN_SECONDS"] = "60"
        reset_llm_failover_state()
        with patch("openai.OpenAI", FakeOpenAI):
            first = chat_completion_with_fallback(
                config=config,
                operation="fallback_probe",
                messages=[{"role": "user", "content": "hello"}],
            )
            second = chat_completion_with_fallback(
                config=config,
                operation="fallback_probe",
                messages=[{"role": "user", "content": "hello again"}],
            )
    finally:
        reset_llm_failover_state()
        for name, value in old_values.items():
            restore_env(name, value)

    assert first.model == second.model == "fallback-model"
    assert first.backend == second.backend == "backup"
    assert calls == [
        ("https://primary.example/v1", "primary-model"),
        ("https://fallback.example/v1", "fallback-model"),
        ("https://fallback.example/v1", "fallback-model"),
    ]


def operation_metrics(operation: str) -> dict[str, int | float]:
    return model_api_metrics_snapshot()["operations"].get(
        operation,
        {
            "calls_total": 0,
            "attempts_total": 0,
            "retries_total": 0,
            "successes_total": 0,
            "failures_total": 0,
            "latency_total_ms": 0.0,
            "latency_max_ms": 0.0,
            "latency_avg_ms": 0.0,
        },
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
