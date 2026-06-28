from __future__ import annotations

import os
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.artifacts import (  # noqa: E402
    ArtifactTenantConflictError,
    delete_metadata_artifact,
    list_metadata_artifacts,
    load_metadata_artifact,
    MindMapArtifact,
    save_metadata_artifact,
)
from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.text_utils import now_ms  # noqa: E402


PREFIX = f"artifact-tenant-smoke-{uuid.uuid4().hex[:10]}"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = replace(
            load_config(),
            metadata_database_url=os.environ.get("SMOKE_METADATA_DATABASE_URL") or None,
            object_store_dir=Path(tmp) / "object_store",
            runtime_dir=Path(tmp) / "runtime",
        )
        with connect_metadata_db(config):
            pass
        test_concurrent_cross_tenant_id_collision(config)
        test_parallel_tenant_lists_remain_isolated(config)
    print("smoke_artifact_multi_tenant_persistence=ok")


def test_concurrent_cross_tenant_id_collision(config) -> None:
    artifact_id = f"{PREFIX}-collision"
    tenants = [f"{PREFIX}-tenant-a", f"{PREFIX}-tenant-b"]
    barrier = threading.Barrier(2)

    def save(tenant_id: str) -> str:
        barrier.wait(timeout=5)
        try:
            save_metadata_artifact(
                config,
                make_artifact(tenant_id=tenant_id, artifact_id=artifact_id),
            )
            return "saved"
        except ArtifactTenantConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save, tenants))
    assert sorted(results) == ["conflict", "saved"]
    owners = [
        tenant_id
        for tenant_id in tenants
        if load_metadata_artifact(
            config,
            tenant_id=tenant_id,
            artifact_id=artifact_id,
        )
        is not None
    ]
    assert len(owners) == 1
    owner = owners[0]
    other = next(tenant_id for tenant_id in tenants if tenant_id != owner)
    original = load_metadata_artifact(config, tenant_id=owner, artifact_id=artifact_id)
    assert original is not None
    save_metadata_artifact(
        config,
        replace(original, title="owner update", updated_at=now_ms()),
    )
    assert load_metadata_artifact(config, tenant_id=owner, artifact_id=artifact_id).title == "owner update"
    assert load_metadata_artifact(config, tenant_id=other, artifact_id=artifact_id) is None
    assert delete_metadata_artifact(config, tenant_id=other, artifact_id=artifact_id) is False
    assert delete_metadata_artifact(config, tenant_id=owner, artifact_id=artifact_id) is True


def test_parallel_tenant_lists_remain_isolated(config) -> None:
    tenants = [f"{PREFIX}-list-{index}" for index in range(12)]

    def save(tenant_id: str) -> None:
        save_metadata_artifact(
            config,
            make_artifact(
                tenant_id=tenant_id,
                artifact_id=f"{tenant_id}-artifact",
            ),
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(save, tenants))
    for tenant_id in tenants:
        rows = list_metadata_artifacts(config, tenant_id=tenant_id)
        assert [row.id for row in rows] == [f"{tenant_id}-artifact"]
        assert all(row.tenant_id == tenant_id for row in rows)
        assert delete_metadata_artifact(
            config,
            tenant_id=tenant_id,
            artifact_id=f"{tenant_id}-artifact",
        )


def make_artifact(*, tenant_id: str, artifact_id: str) -> MindMapArtifact:
    timestamp = now_ms()
    return MindMapArtifact(
        id=artifact_id,
        title=f"synthetic artifact for {tenant_id}",
        status="ready",
        tenant_id=tenant_id,
        workspace_id="workspace-a",
        source_doc_ids=["synthetic-source"],
        created_at=timestamp,
        updated_at=timestamp,
        artifact_type="table",
        table={
            "title": "synthetic",
            "columns": ["field"],
            "rows": [["value"]],
            "summary": "",
        },
    )


if __name__ == "__main__":
    main()
