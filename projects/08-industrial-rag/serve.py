from __future__ import annotations

from pydantic import BaseModel, Field

from answer import answer_query
from rag_core.auth import build_auth_context, validate_bearer_token
from rag_core.config import load_config
from rag_core.events import append_event, hit_event_summaries
from rag_core.pipeline import retrieve_and_rerank
from rag_core.readiness import readiness_report


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    history: list[str] = Field(default_factory=list)
    tenant_id: str = "team_a"
    acl_groups: list[str] = Field(default_factory=list)
    doc_version: int | None = None
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


def create_app():
    from fastapi import FastAPI, Header, HTTPException

    app = FastAPI(title="Industrial RAG Demo", version="0.1.0")

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
        result = retrieve_and_rerank(
            request.query,
            tenant_id=auth_context.tenant_id,
            candidate_limit=request.candidate_limit,
            context_limit=request.context_limit,
            acl_groups=auth_context.acl_groups or None,
            doc_version=request.doc_version,
            source_types=request.source_types or None,
            history=request.history,
            request_id=request.request_id,
        )
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
                "history_len": len(request.history),
                "doc_version": request.doc_version,
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
        result = answer_query(
            request.query,
            tenant_id=auth_context.tenant_id,
            candidate_limit=request.candidate_limit,
            context_limit=request.context_limit,
            acl_groups=auth_context.acl_groups or None,
            doc_version=request.doc_version,
            source_types=request.source_types or None,
            history=request.history,
            request_id=request.request_id,
        )
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
                "history_len": len(request.history),
                "auth_context": auth_context.summary(),
                "doc_version": request.doc_version,
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
    )


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="127.0.0.1", port=8008, reload=False)
