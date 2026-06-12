from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from answer import answer_query
from answer_multimodal import answer_multimodal_query
from rag_core.artifacts import (
    create_mindmap_artifact,
    delete_artifact,
    list_artifacts,
    load_artifact,
)
from rag_core.auth import build_auth_context, validate_bearer_token
from rag_core.config import load_config
from rag_core.events import append_event, hit_event_summaries
from rag_core.pipeline import retrieve_and_rerank
from rag_core.readiness import readiness_report
from rag_core.sources import delete_source, get_source, ingest_uploaded_path, list_sources, save_uploaded_file
from search_multimodal import retrieve_multimodal


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    query_mode: str = Field(default="text", pattern="^(text|multimodal)$")
    history: list[str] = Field(default_factory=list)
    tenant_id: str = "team_a"
    acl_groups: list[str] = Field(default_factory=list)
    doc_version: int | None = None
    doc_ids: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    candidate_limit: int = Field(default=20, ge=1, le=100)
    context_limit: int = Field(default=5, ge=1, le=20)
    request_id: str | None = None


class SearchRequest(QueryRequest):
    pass


class HitResponse(BaseModel):
    doc_id: str
    title: str
    source_uri: str
    source_type: str
    chunk_index: int
    score: float
    rerank_score: float | None = None
    acl_groups: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    text_preview: str = ""


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[HitResponse]
    trace: dict[str, object] | None = None


class SearchResponse(BaseModel):
    request_id: str
    hits: list[HitResponse]
    trace: dict[str, object]


class FeedbackRequest(BaseModel):
    request_id: str
    rating: int = Field(ge=-1, le=1)
    comment: str = ""
    selected_doc_ids: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    status: str
    request_id: str


class SourceResponse(BaseModel):
    doc_id: str
    title: str
    source_type: str
    source_uri: str
    doc_version: int
    chunk_count: int
    acl_groups: list[str]
    status: str
    current: bool
    created_at: int | None = None
    updated_at: int | None = None


class SourceListResponse(BaseModel):
    sources: list[SourceResponse]


class SourceUploadResponse(BaseModel):
    status: str
    sources: list[SourceResponse]
    document_count: int
    chunk_count: int


class DeleteSourceResponse(BaseModel):
    status: str
    doc_id: str
    detail: dict[str, object]


class MindMapRequest(BaseModel):
    title: str = "思维导图"
    tenant_id: str = "team_a"
    acl_groups: list[str] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)
    doc_version: int | None = None


class MindMapArtifactResponse(BaseModel):
    id: str
    title: str
    status: str
    tenant_id: str
    source_doc_ids: list[str]
    created_at: int
    updated_at: int
    root: dict[str, Any] | None = None
    error: str = ""


class ArtifactListResponse(BaseModel):
    artifacts: list[MindMapArtifactResponse]


class DeleteArtifactResponse(BaseModel):
    status: str
    artifact_id: str


def create_app():
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

    app = FastAPI(title="Production RAG", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, object]:
        config = load_config()
        report = readiness_report(config)
        if report["status"] != "ok":
            raise HTTPException(status_code=503, detail=report)
        return report

    @app.get("/sources", response_model=SourceListResponse)
    def sources(
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> SourceListResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        return SourceListResponse(
            sources=[
                source_to_response(source)
                for source in list_sources(config=config, tenant_id=auth_context.tenant_id)
            ]
        )

    @app.post("/sources/upload", response_model=SourceUploadResponse)
    def upload_source(
        file: UploadFile = File(...),
        tenant_id: str = Form("team_a"),
        acl_groups: str = Form("engineering"),
        doc_version: int | None = Form(default=None),
        language: str = Form("zh"),
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> SourceUploadResponse:
        config = load_config()
        body_acl_groups = [item.strip() for item in acl_groups.split(",") if item.strip()]
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=body_acl_groups,
        )
        try:
            saved_path = save_uploaded_file(
                config=config,
                tenant_id=auth_context.tenant_id,
                filename=file.filename or "upload.txt",
                content=file.file,
            )
            summary = ingest_uploaded_path(
                config=config,
                path=saved_path,
                tenant_id=auth_context.tenant_id,
                acl_groups=auth_context.acl_groups or body_acl_groups or ["engineering"],
                doc_version=doc_version,
                language=language,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SourceUploadResponse(
            status="ready",
            sources=[source_to_response(source) for source in summary.sources],
            document_count=summary.document_count,
            chunk_count=summary.chunk_count,
        )

    @app.get("/sources/{doc_id:path}", response_model=SourceResponse)
    def source_detail(
        doc_id: str,
        tenant_id: str = "team_a",
        doc_version: int | None = None,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> SourceResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        source = get_source(
            config=config,
            tenant_id=auth_context.tenant_id,
            doc_id=doc_id,
            doc_version=doc_version,
        )
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        return source_to_response(source)

    @app.delete("/sources/{doc_id:path}", response_model=DeleteSourceResponse)
    def remove_source(
        doc_id: str,
        tenant_id: str = "team_a",
        doc_version: int | None = None,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> DeleteSourceResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        detail = delete_source(
            config=config,
            tenant_id=auth_context.tenant_id,
            doc_id=doc_id,
            doc_version=doc_version,
        )
        return DeleteSourceResponse(status="deleted", doc_id=doc_id, detail=detail)

    @app.post("/search", response_model=SearchResponse)
    def search(
        request: SearchRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> SearchResponse:
        config = load_config()
        auth_context = resolve_auth_context(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            request=request,
        )
        result = resolve_search_result(request, auth_context)
        response = SearchResponse(
            request_id=result.request_id,
            hits=[hit_to_response(hit) for hit in result.hits],
            trace=result.trace.__dict__,
        )
        append_event(
            config.runtime_dir,
            "retrieval_events",
            {
                "request_id": result.request_id,
                "query": request.query,
                "query_mode": request.query_mode,
                "history_len": len(request.history),
                "doc_version": request.doc_version,
                "doc_ids": request.doc_ids,
                "source_types": request.source_types,
                "auth_context": auth_context.summary(),
                "trace": result.trace,
                "raw_hits": hit_event_summaries(result.candidates),
                "rerank_hits": hit_event_summaries(result.reranked),
                "final_context": hit_event_summaries(result.hits),
            },
        )
        return response

    @app.post("/query", response_model=QueryResponse)
    def query(
        request: QueryRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> QueryResponse:
        config = load_config()
        auth_context = resolve_auth_context(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            request=request,
        )
        result = resolve_answer_result(request, auth_context)
        response = QueryResponse(
            request_id=result.request_id,
            answer=result.answer,
            citations=[hit_to_response(hit) for hit in result.hits],
            trace=result.trace.__dict__,
        )
        append_event(
            config.runtime_dir,
            "answer_events",
            {
                "request_id": result.request_id,
                "query": request.query,
                "query_mode": request.query_mode,
                "history_len": len(request.history),
                "auth_context": auth_context.summary(),
                "doc_version": request.doc_version,
                "doc_ids": request.doc_ids,
                "source_types": request.source_types,
                "trace": result.trace,
                "raw_hits": hit_event_summaries(result.candidates),
                "rerank_hits": hit_event_summaries(result.reranked),
                "final_context": hit_event_summaries(result.hits),
                "llm": result.generation,
            },
        )
        return response

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        config = load_config()
        append_event(
            config.runtime_dir,
            "feedback_events",
            request.model_dump(),
        )
        return FeedbackResponse(
            status="accepted",
            request_id=request.request_id,
        )

    @app.get("/artifacts", response_model=ArtifactListResponse)
    def artifacts(
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> ArtifactListResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        return ArtifactListResponse(
            artifacts=[
                artifact_to_response(artifact)
                for artifact in list_artifacts(config, tenant_id=auth_context.tenant_id)
            ]
        )

    @app.post("/artifacts/mindmap", response_model=MindMapArtifactResponse)
    def create_mindmap(
        request: MindMapRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> MindMapArtifactResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=request.tenant_id,
            acl_groups=request.acl_groups,
        )
        artifact = create_mindmap_artifact(
            config,
            title=request.title,
            tenant_id=auth_context.tenant_id,
            source_doc_ids=request.source_doc_ids,
            acl_groups=auth_context.acl_groups or request.acl_groups or None,
            doc_version=request.doc_version,
        )
        return artifact_to_response(artifact)

    @app.get("/artifacts/{artifact_id}", response_model=MindMapArtifactResponse)
    def get_artifact(
        artifact_id: str,
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> MindMapArtifactResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        artifact = load_artifact(config, tenant_id=auth_context.tenant_id, artifact_id=artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return artifact_to_response(artifact)

    @app.delete("/artifacts/{artifact_id}", response_model=DeleteArtifactResponse)
    def remove_artifact(
        artifact_id: str,
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> DeleteArtifactResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        removed = delete_artifact(config, tenant_id=auth_context.tenant_id, artifact_id=artifact_id)
        return DeleteArtifactResponse(
            status="deleted" if removed else "not_found",
            artifact_id=artifact_id,
        )

    return app


def resolve_auth_context(
    *,
    config,
    authorization: str | None,
    x_rag_tenant_id: str | None,
    x_rag_acl_groups: str | None,
    request: QueryRequest,
):
    from fastapi import HTTPException

    try:
        validate_bearer_token(config=config, authorization=authorization)
        return build_auth_context(
            config=config,
            header_tenant_id=x_rag_tenant_id,
            header_acl_groups=x_rag_acl_groups,
            body_tenant_id=request.tenant_id,
            body_acl_groups=request.acl_groups,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def resolve_auth_context_from_values(
    *,
    config,
    authorization: str | None,
    x_rag_tenant_id: str | None,
    x_rag_acl_groups: str | None,
    tenant_id: str,
    acl_groups: list[str],
):
    from fastapi import HTTPException

    try:
        validate_bearer_token(config=config, authorization=authorization)
        return build_auth_context(
            config=config,
            header_tenant_id=x_rag_tenant_id,
            header_acl_groups=x_rag_acl_groups,
            body_tenant_id=tenant_id,
            body_acl_groups=acl_groups,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def resolve_search_result(request: SearchRequest, auth_context):
    if request.query_mode == "multimodal":
        return retrieve_multimodal(
            request.query,
            tenant_id=auth_context.tenant_id,
            candidate_limit=request.candidate_limit,
            context_limit=request.context_limit,
            acl_groups=auth_context.acl_groups or None,
            doc_version=request.doc_version,
            doc_ids=request.doc_ids or None,
            source_types=request.source_types or None,
            history=request.history,
            request_id=request.request_id,
        )
    return retrieve_and_rerank(
        request.query,
        tenant_id=auth_context.tenant_id,
        candidate_limit=request.candidate_limit,
        context_limit=request.context_limit,
        acl_groups=auth_context.acl_groups or None,
        doc_version=request.doc_version,
        doc_ids=request.doc_ids or None,
        source_types=request.source_types or None,
        history=request.history,
        request_id=request.request_id,
    )


def resolve_answer_result(request: QueryRequest, auth_context):
    if request.query_mode == "multimodal":
        return answer_multimodal_query(
            request.query,
            tenant_id=auth_context.tenant_id,
            candidate_limit=request.candidate_limit,
            context_limit=request.context_limit,
            acl_groups=auth_context.acl_groups or None,
            doc_version=request.doc_version,
            doc_ids=request.doc_ids or None,
            source_types=request.source_types or None,
            history=request.history,
            request_id=request.request_id,
        )
    return answer_query(
        request.query,
        tenant_id=auth_context.tenant_id,
        candidate_limit=request.candidate_limit,
        context_limit=request.context_limit,
        acl_groups=auth_context.acl_groups or None,
        doc_version=request.doc_version,
        doc_ids=request.doc_ids or None,
        source_types=request.source_types or None,
        history=request.history,
        request_id=request.request_id,
    )


def hit_to_response(hit) -> HitResponse:
    return HitResponse(
        doc_id=hit.doc_id,
        title=hit.title,
        source_uri=hit.source_uri,
        source_type=hit.source_type,
        chunk_index=hit.chunk_index,
        score=hit.score,
        rerank_score=hit.rerank_score,
        acl_groups=hit.acl_groups,
        metadata=hit.metadata,
        text_preview=hit.text[:360],
    )


def source_to_response(source) -> SourceResponse:
    return SourceResponse(
        doc_id=source.doc_id,
        title=source.title,
        source_type=source.source_type,
        source_uri=source.source_uri,
        doc_version=source.doc_version,
        chunk_count=source.chunk_count,
        acl_groups=source.acl_groups,
        status=source.status,
        current=source.current,
        created_at=source.created_at,
        updated_at=source.updated_at,
    )


def artifact_to_response(artifact) -> MindMapArtifactResponse:
    return MindMapArtifactResponse(
        id=artifact.id,
        title=artifact.title,
        status=artifact.status,
        tenant_id=artifact.tenant_id,
        source_doc_ids=artifact.source_doc_ids,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
        root=artifact.root,
        error=artifact.error,
    )


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="127.0.0.1", port=8008, reload=False)
