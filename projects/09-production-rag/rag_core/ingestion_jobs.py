from __future__ import annotations

import os
import threading
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


_RUNNER_LOCK = threading.Lock()
_RUNNER: IngestionJobRunner | None = None


class IngestionJobRunner:
    def __init__(self, *, workers: int, queue_limit: int, tenant_queue_limit: int | None = None) -> None:
        self.workers = max(1, workers)
        self.queue_limit = max(self.workers, queue_limit)
        self.tenant_queue_limit = max(1, tenant_queue_limit or self.queue_limit)
        self.processing_stale_ms = env_int("RAG_INGEST_PROCESSING_STALE_MS", 30 * 60 * 1000)
        self._slots = threading.BoundedSemaphore(self.queue_limit)
        self._tenant_slots: dict[str, threading.BoundedSemaphore] = {}
        self._tenant_slots_lock = threading.Lock()
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
        if not self._slots.acquire(blocking=False):
            return False
        tenant_slot = self._tenant_slot(tenant_id)
        if not tenant_slot.acquire(blocking=False):
            self._slots.release()
            return False
        claimed = claim_source_task_for_processing(
            config=load_config(),
            tenant_id=tenant_id,
            source=pending_source,
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
            scheduled = False
            try:
                for record in list_queued_source_tasks(config=config, limit=self.queue_limit * 4):
                    tenant_slot = self._tenant_slot(record.tenant_id)
                    if not tenant_slot.acquire(blocking=False):
                        continue
                    claimed = claim_source_task_for_processing(
                        config=config,
                        tenant_id=record.tenant_id,
                        source=record.source,
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
    ) -> None:
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
            delete_source_task(config=config, tenant_id=tenant_id, task_id=source.doc_id)
        except Exception as exc:  # noqa: BLE001 - background job must persist failures.
            config = load_config()
            fail_source_task(config=config, tenant_id=tenant_id, source=source, error=str(exc))
        finally:
            tenant_slot.release()
            self._slots.release()
            self.drain_pending()


def ingestion_job_runner() -> IngestionJobRunner:
    global _RUNNER
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = IngestionJobRunner(
                    workers=env_int("RAG_INGEST_WORKERS", 2),
                    queue_limit=env_int("RAG_INGEST_QUEUE_LIMIT", 32),
                    tenant_queue_limit=env_int("RAG_INGEST_TENANT_QUEUE_LIMIT", 8),
                )
                _RUNNER.drain_pending()
    return _RUNNER


def submit_upload_ingestion_job(
    *,
    pending_source: SourceSummary,
    saved_path: Path,
    tenant_id: str,
    acl_groups: list[str],
    doc_version: int | None,
    language: str,
) -> bool:
    return ingestion_job_runner().submit_upload(
        pending_source=pending_source,
        saved_path=saved_path,
        tenant_id=tenant_id,
        acl_groups=acl_groups,
        doc_version=doc_version,
        language=language,
    )


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        return max(1, int(value))
    except ValueError:
        return default
