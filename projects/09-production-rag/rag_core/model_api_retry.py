from __future__ import annotations

import os
import re
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable
from typing import TypeVar


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


def call_model_api_with_retries(operation: str, func: Callable[[], T]) -> T:
    with model_api_slot(operation):
        attempts = env_int("RAG_MODEL_API_RETRIES", 3)
        base_delay = env_float("RAG_MODEL_API_BACKOFF_SECONDS", 0.5)
        max_delay = env_float("RAG_MODEL_API_BACKOFF_MAX_SECONDS", 8.0)
        for attempt in range(1, attempts + 1):
            try:
                return func()
            except Exception as exc:
                if attempt >= attempts or not is_transient_model_api_error(exc):
                    raise
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                time.sleep(max(0.0, delay))
        raise RuntimeError(f"{operation} failed after {attempts} attempts")


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


def model_api_metrics_snapshot() -> dict[str, int]:
    with _MODEL_API_METRICS_LOCK:
        return dict(_MODEL_API_METRICS)


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
