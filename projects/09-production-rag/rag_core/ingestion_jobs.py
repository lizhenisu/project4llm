from __future__ import annotations

import os
import socket
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag_core.config import load_config
from rag_core.sources import claim_source_task_for_processing
from rag_core.sources import SourceSummary
from rag_core.sources import delete_source_task
from rag_core.sources import fail_source_task
from rag_core.sources import ingest_uploaded_path
from rag_core.sources import list_queued_source_tasks
from rag_core.sources import requeue_stale_processing_source_tasks
from rag_core.sources import renew_source_task_lease


_RUNNER_LOCK = threading.Lock()
_RUNNER: IngestionJobRunner | None = None


class IngestionJobRunner:
    def __init__(
        self,
        *,
        workers: int,
        queue_limit: int,
        tenant_queue_limit: int | None = None,
        runner_id: str | None = None,
    ) -> None:
        self.workers = max(1, workers)
        self.queue_limit = max(self.workers, queue_limit)
        self.tenant_queue_limit = max(1, tenant_queue_limit or self.queue_limit)
        self.processing_stale_ms = env_int("RAG_INGEST_PROCESSING_STALE_MS", 30 * 60 * 1000)
        self.runner_id = runner_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._slots = threading.BoundedSemaphore(self.queue_limit)
        self._tenant_slots: dict[str, threading.BoundedSemaphore] = {}
        self._tenant_slots_lock = threading.Lock()
        self._stopping = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="rag-ingest")

    def submit_upload(
        self,
        *,
        pending_source: SourceSummary,
        saved_path: Path,
        tenant_id: str,
        acl_groups: list[str],
        doc_version: int | None,
        language: str,
    ) -> bool:
        if self._stopping.is_set():
            return False
        if not self._slots.acquire(blocking=False):
            return False
        tenant_slot = self._tenant_slot(tenant_id)
        if not tenant_slot.acquire(blocking=False):
            self._slots.release()
            return False
        lease_owner = self._lease_owner(pending_source.doc_id)
        claimed = claim_source_task_for_processing(
            config=load_config(),
            tenant_id=tenant_id,
            source=pending_source,
            lease_owner=lease_owner,
            lease_ms=self.processing_stale_ms,
        )
        if claimed is None:
            tenant_slot.release()
            self._slots.release()
            return True
        self._executor.submit(
            self._run_claimed_upload,
            claimed,
            tenant_id,
            tenant_slot,
            language,
            lease_owner,
        )
        return True

    def _tenant_slot(self, tenant_id: str) -> threading.BoundedSemaphore:
        with self._tenant_slots_lock:
            slot = self._tenant_slots.get(tenant_id)
            if slot is None:
                slot = threading.BoundedSemaphore(self.tenant_queue_limit)
                self._tenant_slots[tenant_id] = slot
            return slot

    def drain_pending(self) -> None:
        if self._stopping.is_set():
            return
        config = load_config()
        try:
            requeue_stale_processing_source_tasks(
                config=config,
                stale_after_ms=self.processing_stale_ms,
                limit=self.queue_limit * 4,
            )
        except Exception:
            pass
        while self._slots.acquire(blocking=False):
            if self._stopping.is_set():
                self._slots.release()
                return
            scheduled = False
            try:
                for record in list_queued_source_tasks(config=config, limit=self.queue_limit * 4):
                    tenant_slot = self._tenant_slot(record.tenant_id)
                    if not tenant_slot.acquire(blocking=False):
                        continue
                    lease_owner = self._lease_owner(record.source.doc_id)
                    claimed = claim_source_task_for_processing(
                        config=config,
                        tenant_id=record.tenant_id,
                        source=record.source,
                        lease_owner=lease_owner,
                        lease_ms=self.processing_stale_ms,
                    )
                    if claimed is None:
                        tenant_slot.release()
                        continue
                    self._executor.submit(
                        self._run_claimed_upload,
                        claimed,
                        record.tenant_id,
                        tenant_slot,
                        "zh",
                        lease_owner,
                    )
                    scheduled = True
                    break
            except Exception:
                scheduled = False
            if not scheduled:
                self._slots.release()
                return

    def _run_claimed_upload(
        self,
        source: SourceSummary,
        tenant_id: str,
        tenant_slot: threading.BoundedSemaphore,
        language: str,
        lease_owner: str,
    ) -> None:
        stop_renewal = threading.Event()
        lease_valid = threading.Event()
        lease_valid.set()
        renewal_thread = threading.Thread(
            target=self._renew_lease_loop,
            args=(source.doc_id, tenant_id, lease_owner, stop_renewal, lease_valid),
            name=f"rag-ingest-lease-{source.doc_id}",
            daemon=True,
        )
        renewal_thread.start()
        try:
            config = load_config()
            ingest_uploaded_path(
                config=config,
                path=Path(source.source_uri),
                tenant_id=tenant_id,
                acl_groups=source.acl_groups,
                doc_version=source.doc_version,
                language=language,
            )
            if lease_valid.is_set():
                delete_source_task(
                    config=config,
                    tenant_id=tenant_id,
                    task_id=source.doc_id,
                    lease_owner=lease_owner,
                )
        except Exception as exc:  # noqa: BLE001 - background job must persist failures.
            config = load_config()
            if lease_valid.is_set():
                fail_source_task(
                    config=config,
                    tenant_id=tenant_id,
                    source=source,
                    error=str(exc),
                    lease_owner=lease_owner,
                )
        finally:
            stop_renewal.set()
            renewal_thread.join(timeout=1)
            tenant_slot.release()
            self._slots.release()
            self.drain_pending()

    def _renew_lease_loop(
        self,
        task_id: str,
        tenant_id: str,
        lease_owner: str,
        stop: threading.Event,
        lease_valid: threading.Event,
    ) -> None:
        interval_seconds = max(0.25, self.processing_stale_ms / 3000)
        while not stop.wait(interval_seconds):
            try:
                renewed = renew_source_task_lease(
                    config=load_config(),
                    tenant_id=tenant_id,
                    task_id=task_id,
                    lease_owner=lease_owner,
                    lease_ms=self.processing_stale_ms,
                )
            except Exception:
                renewed = False
            if not renewed:
                lease_valid.clear()
                return

    def _lease_owner(self, task_id: str) -> str:
        return f"{self.runner_id}:{task_id}:{uuid.uuid4().hex}"

    def shutdown(self, *, wait: bool = True) -> None:
        self._stopping.set()
        self._executor.shutdown(wait=wait)


def ingestion_job_runner() -> IngestionJobRunner:
    global _RUNNER
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = new_ingestion_job_runner()
                _RUNNER.drain_pending()
    return _RUNNER


def new_ingestion_job_runner() -> IngestionJobRunner:
    return IngestionJobRunner(
        workers=env_int("RAG_INGEST_WORKERS", 2),
        queue_limit=env_int("RAG_INGEST_QUEUE_LIMIT", 32),
        tenant_queue_limit=env_int("RAG_INGEST_TENANT_QUEUE_LIMIT", 8),
    )


def run_ingestion_worker(
    *,
    stop_event: threading.Event,
    poll_seconds: float | None = None,
    runner: IngestionJobRunner | None = None,
) -> None:
    active_runner = runner or new_ingestion_job_runner()
    interval = poll_seconds if poll_seconds is not None else env_float("RAG_INGEST_POLL_SECONDS", 1.0)
    interval = max(0.1, interval)
    try:
        active_runner.drain_pending()
        while not stop_event.wait(interval):
            active_runner.drain_pending()
    finally:
        active_runner.shutdown(wait=True)


def submit_upload_ingestion_job(
    *,
    pending_source: SourceSummary,
    saved_path: Path,
    tenant_id: str,
    acl_groups: list[str],
    doc_version: int | None,
    language: str,
) -> bool:
    if ingestion_execution_mode() == "external":
        return True
    return ingestion_job_runner().submit_upload(
        pending_source=pending_source,
        saved_path=saved_path,
        tenant_id=tenant_id,
        acl_groups=acl_groups,
        doc_version=doc_version,
        language=language,
    )


def ingestion_execution_mode() -> str:
    mode = os.environ.get("RAG_INGEST_EXECUTION_MODE", "embedded").strip().lower()
    if mode not in {"embedded", "external"}:
        raise ValueError(
            "RAG_INGEST_EXECUTION_MODE must be 'embedded' or 'external', "
            f"got: {mode or '<empty>'}"
        )
    return mode


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        return max(1, int(value))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, str(default))
    try:
        return max(0.1, float(value))
    except ValueError:
        return default
