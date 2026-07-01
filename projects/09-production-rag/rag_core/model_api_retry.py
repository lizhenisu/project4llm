from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar


T = TypeVar("T")

TRANSIENT_STATUS_CODES = {408, 409, 425, 429}
_MODEL_API_SEMAPHORE_LOCK = threading.Lock()
_MODEL_API_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}
_MODEL_API_METRICS_LOCK = threading.Lock()
_MODEL_API_METRICS = {
    "active": 0,
    "acquired_total": 0,
    "rejected_total": 0,
}
_MODEL_API_OPERATION_METRICS: dict[str, dict[str, int | float]] = {}
_LLM_FAILOVER_LOCK = threading.Lock()
_LLM_PRIMARY_CIRCUITS: dict[tuple[str, str], float] = {}


@dataclass(frozen=True)
class ChatCompletionResult:
    response: Any
    model: str
    backend: str
    client: Any


def call_model_api_with_retries(
    operation: str,
    func: Callable[[], T],
    *,
    usage_provider: str = "",
    usage_model: str = "",
) -> T:
    operation_key = normalize_operation(operation)
    started = time.perf_counter()
    succeeded = False
    record_operation_call(operation_key)
    try:
        with model_api_slot(operation_key):
            attempts = env_int("RAG_MODEL_API_RETRIES", 3)
            base_delay = env_float("RAG_MODEL_API_BACKOFF_SECONDS", 0.5)
            max_delay = env_float("RAG_MODEL_API_BACKOFF_MAX_SECONDS", 8.0)
            for attempt in range(1, attempts + 1):
                record_operation_attempt(operation_key)
                try:
                    result = func()
                    if usage_provider and usage_model:
                        from rag_core.model_usage import record_response_model_usage

                        record_response_model_usage(
                            response=result,
                            provider=usage_provider,
                            model=usage_model,
                            operation=operation_key,
                        )
                    succeeded = True
                    return result
                except Exception as exc:
                    if attempt >= attempts or not is_transient_model_api_error(exc):
                        raise
                    record_operation_retry(operation_key)
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    time.sleep(max(0.0, delay))
            raise RuntimeError(f"{operation_key} failed after {attempts} attempts")
    finally:
        record_operation_finished(
            operation_key,
            succeeded=succeeded,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def chat_completion_with_fallback(
    *,
    config,
    operation: str,
    messages: list[dict[str, object]],
    **kwargs: object,
) -> ChatCompletionResult:
    from openai import OpenAI

    if not config.llm_base_url or not config.llm_api_key:
        raise RuntimeError("NEW_API_URL/NEW_API_KEY must be configured for model generation.")
    fallback = resolve_llm_fallback(config)
    primary_client = OpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        max_retries=0,
    )
    primary_key = (str(config.llm_base_url), str(config.llm_model))
    if fallback is not None and llm_primary_circuit_open(primary_key):
        return call_fallback_completion(
            fallback=fallback,
            operation=operation,
            messages=messages,
            kwargs=kwargs,
        )
    try:
        response = call_model_api_with_retries(
            operation,
            lambda: primary_client.chat.completions.create(
                model=config.llm_model,
                messages=messages,
                **kwargs,
            ),
            usage_provider="newapi",
            usage_model=config.llm_model,
        )
    except Exception as exc:
        if fallback is None or not is_transient_model_api_error(exc):
            raise
        open_llm_primary_circuit(primary_key)
        return call_fallback_completion(
            fallback=fallback,
            operation=operation,
            messages=messages,
            kwargs=kwargs,
        )
    close_llm_primary_circuit(primary_key)
    return ChatCompletionResult(
        response=response,
        model=config.llm_model,
        backend="newapi",
        client=primary_client,
    )


def call_fallback_completion(
    *,
    fallback: tuple[str, str, str, str],
    operation: str,
    messages: list[dict[str, object]],
    kwargs: dict[str, object],
) -> ChatCompletionResult:
    from openai import OpenAI

    fallback_base_url, fallback_api_key, fallback_model, fallback_backend = fallback
    fallback_client = OpenAI(
        base_url=fallback_base_url,
        api_key=fallback_api_key,
        max_retries=0,
    )
    response = call_model_api_with_retries(
        f"{operation}_fallback",
        lambda: fallback_client.chat.completions.create(
            model=fallback_model,
            messages=messages,
            **kwargs,
        ),
        usage_provider=fallback_backend,
        usage_model=fallback_model,
    )
    return ChatCompletionResult(
        response=response,
        model=fallback_model,
        backend=fallback_backend,
        client=fallback_client,
    )


def llm_primary_circuit_open(key: tuple[str, str]) -> bool:
    now = time.monotonic()
    with _LLM_FAILOVER_LOCK:
        expires_at = _LLM_PRIMARY_CIRCUITS.get(key, 0.0)
        if expires_at <= now:
            _LLM_PRIMARY_CIRCUITS.pop(key, None)
            return False
        return True


def open_llm_primary_circuit(key: tuple[str, str]) -> None:
    cooldown = env_float("RAG_LLM_FAILOVER_COOLDOWN_SECONDS", 60.0)
    if cooldown <= 0:
        return
    with _LLM_FAILOVER_LOCK:
        _LLM_PRIMARY_CIRCUITS[key] = time.monotonic() + cooldown


def close_llm_primary_circuit(key: tuple[str, str]) -> None:
    with _LLM_FAILOVER_LOCK:
        _LLM_PRIMARY_CIRCUITS.pop(key, None)


def reset_llm_failover_state() -> None:
    with _LLM_FAILOVER_LOCK:
        _LLM_PRIMARY_CIRCUITS.clear()


def resolve_llm_fallback(config) -> tuple[str, str, str, str] | None:
    base_url = normalize_openai_base_url(os.environ.get("RAG_LLM_FALLBACK_BASE_URL", ""))
    model = os.environ.get("RAG_LLM_FALLBACK_MODEL", "").strip()
    if not base_url or not model:
        return None
    api_key = os.environ.get("RAG_LLM_FALLBACK_API_KEY", "").strip()
    siliconflow_base_url = normalize_openai_base_url(
        str(getattr(config, "siliconflow_base_url", "") or "")
    )
    if not api_key and base_url == siliconflow_base_url:
        api_key = str(getattr(config, "siliconflow_api_key", "") or "").strip()
    if not api_key:
        return None
    backend = os.environ.get("RAG_LLM_FALLBACK_BACKEND", "fallback").strip() or "fallback"
    return base_url, api_key, model, backend


def normalize_openai_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    if base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def public_model_api_error(exc: Exception) -> str:
    if is_transient_model_api_error(exc):
        return "大模型服务暂时不可用，请稍后重试。"
    return str(exc) or exc.__class__.__name__


@contextmanager
def model_api_slot(operation: str):
    limit = env_int("RAG_MODEL_API_MAX_CONCURRENCY", 32)
    timeout = env_float("RAG_MODEL_API_QUEUE_TIMEOUT_SECONDS", 30.0)
    semaphore = model_api_semaphore(limit)
    acquired = semaphore.acquire(timeout=timeout)
    if not acquired:
        record_model_api_rejected()
        raise RuntimeError(
            f"Model API concurrency limit reached for {operation}; "
            f"RAG_MODEL_API_MAX_CONCURRENCY={limit}"
        )
    record_model_api_acquired()
    try:
        yield
    finally:
        record_model_api_released()
        semaphore.release()


def model_api_semaphore(limit: int) -> threading.BoundedSemaphore:
    with _MODEL_API_SEMAPHORE_LOCK:
        semaphore = _MODEL_API_SEMAPHORES.get(limit)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _MODEL_API_SEMAPHORES[limit] = semaphore
        return semaphore


def record_model_api_acquired() -> None:
    with _MODEL_API_METRICS_LOCK:
        _MODEL_API_METRICS["active"] += 1
        _MODEL_API_METRICS["acquired_total"] += 1


def record_model_api_released() -> None:
    with _MODEL_API_METRICS_LOCK:
        _MODEL_API_METRICS["active"] = max(0, _MODEL_API_METRICS["active"] - 1)


def record_model_api_rejected() -> None:
    with _MODEL_API_METRICS_LOCK:
        _MODEL_API_METRICS["rejected_total"] += 1


def model_api_metrics_snapshot() -> dict[str, object]:
    with _MODEL_API_METRICS_LOCK:
        operations = {
            operation: {
                **metrics,
                "latency_avg_ms": (
                    round(float(metrics["latency_total_ms"]) / int(metrics["calls_total"]), 2)
                    if int(metrics["calls_total"])
                    else 0.0
                ),
                "latency_total_ms": round(float(metrics["latency_total_ms"]), 2),
                "latency_max_ms": round(float(metrics["latency_max_ms"]), 2),
            }
            for operation, metrics in sorted(_MODEL_API_OPERATION_METRICS.items())
        }
        return {**_MODEL_API_METRICS, "operations": operations}


def normalize_operation(operation: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", operation.strip())[:80]
    return normalized or "unknown"


def new_operation_metrics() -> dict[str, int | float]:
    return {
        "calls_total": 0,
        "attempts_total": 0,
        "retries_total": 0,
        "successes_total": 0,
        "failures_total": 0,
        "latency_total_ms": 0.0,
        "latency_max_ms": 0.0,
    }


def record_operation_call(operation: str) -> None:
    with _MODEL_API_METRICS_LOCK:
        metrics = _MODEL_API_OPERATION_METRICS.setdefault(operation, new_operation_metrics())
        metrics["calls_total"] = int(metrics["calls_total"]) + 1


def record_operation_attempt(operation: str) -> None:
    with _MODEL_API_METRICS_LOCK:
        metrics = _MODEL_API_OPERATION_METRICS.setdefault(operation, new_operation_metrics())
        metrics["attempts_total"] = int(metrics["attempts_total"]) + 1


def record_operation_retry(operation: str) -> None:
    with _MODEL_API_METRICS_LOCK:
        metrics = _MODEL_API_OPERATION_METRICS.setdefault(operation, new_operation_metrics())
        metrics["retries_total"] = int(metrics["retries_total"]) + 1


def record_operation_finished(operation: str, *, succeeded: bool, latency_ms: float) -> None:
    with _MODEL_API_METRICS_LOCK:
        metrics = _MODEL_API_OPERATION_METRICS.setdefault(operation, new_operation_metrics())
        outcome_key = "successes_total" if succeeded else "failures_total"
        metrics[outcome_key] = int(metrics[outcome_key]) + 1
        metrics["latency_total_ms"] = float(metrics["latency_total_ms"]) + latency_ms
        metrics["latency_max_ms"] = max(float(metrics["latency_max_ms"]), latency_ms)


def is_transient_model_api_error(exc: Exception) -> bool:
    status_code = exception_status_code(exc)
    if status_code is not None:
        return status_code in TRANSIENT_STATUS_CODES or status_code >= 500

    message = str(exc)
    if re.search(r"\b(408|409|425|429|5\d\d)\b", message):
        return True
    error_name = exc.__class__.__name__.lower()
    return "timeout" in error_name or "connection" in error_name or "urlerror" in error_name


def exception_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    return None


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        return default
