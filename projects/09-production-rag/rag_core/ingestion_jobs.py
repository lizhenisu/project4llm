from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag_core.config import load_config
from rag_core.sources import SourceSummary
from rag_core.sources import delete_source_task
from rag_core.sources import fail_source_task
from rag_core.sources import ingest_uploaded_path


_RUNNER_LOCK = threading.Lock()
_RUNNER: IngestionJobRunner | None = None


class IngestionJobRunner:
    def __init__(self, *, workers: int, queue_limit: int) -> None:
        self.workers = max(1, workers)
        self.queue_limit = max(self.workers, queue_limit)
        self._slots = threading.BoundedSemaphore(self.queue_limit)
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
        self._executor.submit(
            self._run_upload,
            pending_source,
            saved_path,
            tenant_id,
            acl_groups,
            doc_version,
            language,
        )
        return True

    def _run_upload(
        self,
        pending_source: SourceSummary,
        saved_path: Path,
        tenant_id: str,
        acl_groups: list[str],
        doc_version: int | None,
        language: str,
    ) -> None:
        try:
            config = load_config()
            ingest_uploaded_path(
                config=config,
                path=saved_path,
                tenant_id=tenant_id,
                acl_groups=acl_groups,
                doc_version=doc_version,
                language=language,
            )
            delete_source_task(config=config, tenant_id=tenant_id, task_id=pending_source.doc_id)
        except Exception as exc:  # noqa: BLE001 - background job must persist failures.
            config = load_config()
            fail_source_task(config=config, tenant_id=tenant_id, source=pending_source, error=str(exc))
        finally:
            self._slots.release()


def ingestion_job_runner() -> IngestionJobRunner:
    global _RUNNER
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = IngestionJobRunner(
                    workers=env_int("RAG_INGEST_WORKERS", 2),
                    queue_limit=env_int("RAG_INGEST_QUEUE_LIMIT", 32),
                )
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
