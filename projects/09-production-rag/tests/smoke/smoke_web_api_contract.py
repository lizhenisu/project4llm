from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from serve import (
    ArtifactListResponse,
    DeleteArtifactResponse,
    DeleteSourceResponse,
    HitResponse,
    MindMapArtifactResponse,
    SourceListResponse,
    SourceResponse,
    SourceUploadResponse,
    create_app,
)


def main() -> None:
    app = create_app()
    routes = {route.path for route in app.routes}
    required_routes = {
        "/sources",
        "/sources/upload",
        "/sources/{doc_id:path}",
        "/artifacts",
        "/artifacts/mindmap",
        "/artifacts/{artifact_id}",
    }
    missing = sorted(required_routes - routes)
    assert not missing, missing

    source = SourceResponse(
        doc_id="runbook",
        title="Runbook",
        source_type="md",
        source_uri="memory://runbook",
        doc_version=1,
        chunk_count=2,
        acl_groups=["ops"],
        status="ready",
        current=True,
    )
    assert SourceListResponse(sources=[source]).sources[0].doc_id == "runbook"
    assert (
        SourceUploadResponse(
            status="accepted",
            sources=[source],
            document_count=1,
            chunk_count=2,
        ).sources[0].status
        == "ready"
    )
    assert DeleteSourceResponse(status="deleted", doc_id="runbook", detail={}).status == "deleted"

    hit = HitResponse(
        doc_id="runbook",
        doc_version=1,
        source_type="md",
        source_uri="memory://runbook",
        title="Runbook",
        chunk_index=0,
        text_preview="preview",
        score=0.9,
        rerank_score=0.8,
        acl_groups=["ops"],
        metadata={},
    )
    assert hit.text_preview == "preview"

    artifact = MindMapArtifactResponse(
        id="artifact-1",
        title="Mind Map",
        status="ready",
        tenant_id="team_a",
        source_doc_ids=["runbook"],
        created_at=1,
        updated_at=1,
        root={"id": "root", "label": "Mind Map", "children": []},
    )
    assert ArtifactListResponse(artifacts=[artifact]).artifacts[0].id == "artifact-1"
    assert DeleteArtifactResponse(status="deleted", artifact_id="artifact-1").status == "deleted"

    print("smoke_web_api_contract=ok")


if __name__ == "__main__":
    main()
