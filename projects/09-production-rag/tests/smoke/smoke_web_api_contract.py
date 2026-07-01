from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve
from rag_core.auth import AuthContext
from rag_core.sources import SourceSummary
from rag_core.upload_admission import UploadAdmissionReservation
from serve import (
    ArtifactListResponse,
    DeleteArtifactResponse,
    DeleteConversationResponse,
    DeleteSourceResponse,
    HitResponse,
    MindMapArtifactResponse,
    ConversationListResponse,
    ConversationMessageRequest,
    ConversationResponse,
    SourceListResponse,
    SourceResponse,
    SourceUploadResponse,
    create_app,
)


def main() -> None:
    app = create_app()
    routes = {route.path for route in app.routes}
    methods_by_route: dict[str, set[str]] = {}
    for route in app.routes:
        methods_by_route.setdefault(route.path, set()).update(getattr(route, "methods", set()) or set())
    required_routes = {
        "/sources",
        "/sources/upload",
        "/sources/{doc_id:path}",
        "/conversations",
        "/conversations/{conversation_id}",
        "/artifacts",
        "/artifacts/mindmap",
        "/artifacts/{artifact_id}",
    }
    missing = sorted(required_routes - routes)
    assert not missing, missing
    assert "GET" in methods_by_route["/sources"]
    assert "POST" in methods_by_route["/sources/upload"]
    assert "GET" in methods_by_route["/sources/{doc_id:path}"]
    assert "DELETE" in methods_by_route["/sources/{doc_id:path}"]
    assert "GET" in methods_by_route["/conversations"]
    assert "POST" in methods_by_route["/conversations"]
    assert "GET" in methods_by_route["/conversations/{conversation_id}"]
    assert "PATCH" in methods_by_route["/conversations/{conversation_id}"]
    assert "DELETE" in methods_by_route["/conversations/{conversation_id}"]
    assert "GET" in methods_by_route["/artifacts"]
    assert "POST" in methods_by_route["/artifacts/mindmap"]
    assert "GET" in methods_by_route["/artifacts/{artifact_id}"]
    assert "DELETE" in methods_by_route["/artifacts/{artifact_id}"]

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

    conversation = ConversationResponse(
        id="conv-1",
        tenant_id="team_a",
        title="Runbook chat",
        messages=[
            ConversationMessageRequest(
                id="msg-1",
                role="user",
                content="怎么排障？",
            ),
            ConversationMessageRequest(
                id="msg-2",
                role="assistant",
                content="检查 Milvus。",
                citations=[hit],
            ),
        ],
        source_doc_ids=["runbook"],
        created_at=1,
        updated_at=2,
    )
    assert conversation.messages[1].citations[0].doc_id == "runbook"
    assert ConversationListResponse(conversations=[]).conversations == []
    assert DeleteConversationResponse(status="deleted", conversation_id="conv-1").status == "deleted"

    queued_source = SourceSummary(
        doc_id="runbook",
        title="upload.txt",
        source_type="txt",
        source_uri="/tmp/upload.txt",
        doc_version=1,
        chunk_count=0,
        acl_groups=["ops"],
        status="queued",
        current=False,
        created_at=1,
        updated_at=1,
    )
    upload_reservation = UploadAdmissionReservation(
        owner="web-contract-upload-reservation",
        tenant_id="team_a",
        expires_at=9999999999999,
    )
    with (
        patch("serve.acquire_upload_admission_reservation", return_value=upload_reservation),
        patch("serve.save_uploaded_file", return_value=Path("/tmp/upload.txt")) as save_uploaded,
        patch("serve.create_source_task", return_value=queued_source) as create_task,
        patch("serve.submit_upload_ingestion_job") as submit_job,
        patch("serve.resolve_auth_context_from_values", return_value=AuthContext("team_a", ["ops"], "smoke")),
    ):
        upload_response = TestClient(app).post(
            "/sources/upload",
            files={"file": ("upload.txt", b"hello", "text/plain")},
            data={"tenant_id": "team_a", "acl_groups": "ops"},
        )
    assert upload_response.status_code == 200, upload_response.text
    assert upload_response.json()["status"] == "queued"
    assert upload_response.json()["sources"][0]["doc_id"] == "runbook"
    assert upload_response.json()["sources"][0]["status"] == "queued"
    save_uploaded.assert_called_once()
    create_task.assert_called_once()
    assert create_task.call_args.kwargs["upload_reservation_owner"] == upload_reservation.owner
    submit_job.assert_called_once()

    old_runtime_dir = os.environ.get("RAG_RUNTIME_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["RAG_RUNTIME_DIR"] = tmp
        try:
            client = TestClient(app)
            with patch(
                "serve.resolve_auth_context_from_values",
                return_value=AuthContext("team_a", ["ops"], "smoke"),
            ):
                save_response = client.post(
                    "/conversations",
                    json={
                        "tenant_id": "team_a",
                        "title": "Runbook chat",
                        "source_doc_ids": ["runbook"],
                        "messages": [
                            {"id": "msg-1", "role": "user", "content": "怎么排障？"},
                            {
                                "id": "msg-2",
                                "role": "assistant",
                                "content": "检查 Milvus。",
                                "citations": [hit.model_dump()],
                            },
                        ],
                    },
                )
                assert save_response.status_code == 200, save_response.text
                conversation_id = save_response.json()["id"]
                list_response = client.get("/conversations?tenant_id=team_a")
                assert list_response.status_code == 200, list_response.text
                assert list_response.json()["conversations"][0]["id"] == conversation_id
                get_response = client.get(f"/conversations/{conversation_id}?tenant_id=team_a")
                assert get_response.status_code == 200, get_response.text
                assert get_response.json()["messages"][1]["citations"][0]["doc_id"] == "runbook"
                rename_response = client.patch(
                    f"/conversations/{conversation_id}?tenant_id=team_a",
                    json={"title": "Renamed conversation"},
                )
                assert rename_response.status_code == 200, rename_response.text
                assert rename_response.json()["title"] == "Renamed conversation"
                assert client.get(f"/conversations/{conversation_id}?tenant_id=team_a").json()["title"] == "Renamed conversation"
                delete_response = client.delete(f"/conversations/{conversation_id}?tenant_id=team_a")
                assert delete_response.status_code == 200, delete_response.text
                assert delete_response.json()["status"] == "deleted"
        finally:
            if old_runtime_dir is None:
                os.environ.pop("RAG_RUNTIME_DIR", None)
            else:
                os.environ["RAG_RUNTIME_DIR"] = old_runtime_dir

    print("smoke_web_api_contract=ok")


if __name__ == "__main__":
    main()
