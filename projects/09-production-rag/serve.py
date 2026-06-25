from __future__ import annotations

import time
import uuid
import base64
import json
import mimetypes
import threading
from dataclasses import replace
from queue import Queue
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from answer import answer_query
from answer_multimodal import answer_multimodal_query
from rag_core.artifacts import (
    MindMapArtifact,
    build_llm_table,
    build_mindmap_root,
    delete_artifact,
    delete_metadata_artifact,
    fail_metadata_artifact,
    list_artifacts,
    list_metadata_artifacts,
    load_artifact,
    load_metadata_artifact,
    save_metadata_artifact,
)
from rag_core.auth import build_auth_context, validate_bearer_token
from rag_core.app_version import app_version
from rag_core.config import load_config
from rag_core.conversations import (
    ConversationMessage,
    delete_conversation,
    list_conversations,
    load_conversation,
    save_conversation,
)
from rag_core.events import append_event, hit_event_summaries
from rag_core.ingestion_jobs import submit_upload_ingestion_job
from rag_core.jsonl_store import read_object_bytes_by_uri, unquote_object_uri
from rag_core.pipeline import retrieve_and_rerank
from rag_core.readiness import readiness_report
from rag_core.sources import (
    create_source_task,
    delete_source,
    fail_source_task,
    get_source,
    get_source_content,
    list_sources,
    rename_source,
    resolve_metadata_display_block_urls,
    save_uploaded_file,
)
from rag_core.user_auth import (
    authenticate_token,
    bearer_token,
    bulk_update_users,
    change_user_password,
    count_public_users,
    create_announcement,
    delete_announcement,
    ensure_default_test_account,
    is_registration_enabled,
    list_announcements,
    list_public_users,
    login_user,
    logout_user,
    register_user,
    refresh_session_token,
    set_registration_enabled,
    set_user_status,
    update_user_profile,
)
from search_multimodal import retrieve_multimodal


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    query_mode: str = Field(default="text", pattern="^(text|multimodal)$")
    image_data_url: str | None = None
    history: list[str] = Field(default_factory=list)
    tenant_id: str = "team_a"
    acl_groups: list[str] = Field(default_factory=list)
    doc_version: int | None = None
    doc_ids: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    include_all_sources: bool = False
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
    text: str = ""
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
    tenant_id: str = "team_a"
    acl_groups: list[str] = Field(default_factory=list)


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
    child_doc_ids: list[str] = Field(default_factory=list)
    error: str = ""


class SourceListResponse(BaseModel):
    sources: list[SourceResponse]


class SourceUploadResponse(BaseModel):
    status: str
    sources: list[SourceResponse]
    document_count: int
    chunk_count: int


class SourceContentResponse(BaseModel):
    doc_id: str
    title: str
    source_type: str
    source_uri: str
    doc_version: int
    child_doc_ids: list[str]
    guide: str
    tags: list[str]
    text: str
    blocks: list[dict[str, str]] = Field(default_factory=list)
    suggested_title: str = ""


class DeleteSourceResponse(BaseModel):
    status: str
    doc_id: str
    detail: dict[str, object]


class RenameSourceRequest(BaseModel):
    title: str


class RenameSourceResponse(BaseModel):
    status: str
    doc_id: str
    title: str


class MindMapRequest(BaseModel):
    title: str = "思维导图"
    tenant_id: str = "team_a"
    workspace_id: str = ""
    acl_groups: list[str] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)
    doc_version: int | None = None
    context_limit: int = Field(default=5, ge=1, le=20)


class MindMapArtifactResponse(BaseModel):
    id: str
    title: str
    status: str
    tenant_id: str
    workspace_id: str = ""
    source_doc_ids: list[str]
    created_at: int
    updated_at: int
    artifact_type: str = "mindmap"
    root: dict[str, Any] | None = None
    table: dict[str, Any] | None = None
    error: str = ""


class ArtifactListResponse(BaseModel):
    artifacts: list[MindMapArtifactResponse]


class DeleteArtifactResponse(BaseModel):
    status: str
    artifact_id: str


class RenameArtifactRequest(BaseModel):
    title: str


class RenameArtifactResponse(BaseModel):
    status: str
    artifact_id: str
    title: str


class ConversationMessageRequest(BaseModel):
    id: str
    role: str = Field(pattern="^(user|assistant)$")
    content: str
    status: str = Field(default="done", pattern="^(sending|done|failed)$")
    request_id: str | None = None
    citations: list[HitResponse] = Field(default_factory=list)
    image_data_url: str | None = None
    created_at: int | None = None
    feedback_rating: int | None = Field(default=None, ge=-1, le=1)
    rag_progress: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("rag_progress", mode="before")
    @classmethod
    def normalize_rag_progress(cls, value):
        return [] if value is None else value


class ConversationUpsertRequest(BaseModel):
    id: str | None = None
    tenant_id: str = "team_a"
    title: str = ""
    messages: list[ConversationMessageRequest]
    source_doc_ids: list[str] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    id: str
    tenant_id: str
    title: str
    messages: list[ConversationMessageRequest]
    source_doc_ids: list[str]
    created_at: int
    updated_at: int


class ConversationListItemResponse(BaseModel):
    id: str
    tenant_id: str
    title: str
    message_count: int
    source_doc_ids: list[str]
    created_at: int
    updated_at: int


class ConversationListResponse(BaseModel):
    conversations: list[ConversationListItemResponse]


class DeleteConversationResponse(BaseModel):
    status: str
    conversation_id: str


class UserResponse(BaseModel):
    id: str
    username: str
    display_name: str
    role: str
    tenant_id: str
    created_at: int
    avatar_url: str = ""
    status: str = "active"
    profile_name_edit_allowed: bool = True
    avatar_edit_allowed: bool = True
    last_login_at: int | None = None


class AuthRequest(BaseModel):
    username: str
    password: str
    display_name: str | None = None


class AuthResponse(BaseModel):
    user: UserResponse
    token: str
    expires_at: int


class ProfileUpdateRequest(BaseModel):
    username: str
    display_name: str
    avatar_url: str = ""


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class UserStatusRequest(BaseModel):
    status: str = Field(pattern="^(active|banned)$")


class AdminUserUpdateItem(BaseModel):
    user_id: str
    status: str | None = Field(default=None, pattern="^(active|banned)$")
    profile_name_edit_allowed: bool | None = None
    avatar_edit_allowed: bool | None = None


class AdminUserBulkUpdateRequest(BaseModel):
    users: list[AdminUserUpdateItem] = Field(default_factory=list, min_length=1, max_length=50)


class AnnouncementRequest(BaseModel):
    title: str
    content: str
    link_url: str = ""
    link_label: str = ""


class AnnouncementResponse(BaseModel):
    id: str
    title: str
    content: str
    link_url: str = ""
    link_label: str = ""
    author_id: str
    author_name: str | None = None
    created_at: int


class AnnouncementListResponse(BaseModel):
    announcements: list[AnnouncementResponse]


class AdminSettingsResponse(BaseModel):
    registration_enabled: bool
    latest_announcement: AnnouncementResponse | None = None


class RegistrationSettingsRequest(BaseModel):
    registration_enabled: bool


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total: int
    limit: int
    offset: int
    query: str = ""


def create_app():
    app = FastAPI(title="Production RAG", version=app_version())
    ensure_default_test_account(load_config())

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

    @app.post("/auth/register", response_model=AuthResponse)
    def register(request: AuthRequest) -> AuthResponse:
        config = load_config()
        try:
            user = register_user(
                config,
                username=request.username,
                password=request.password,
                display_name=request.display_name,
            )
            user, token, expires_at = login_user(
                config,
                username=request.username,
                password=request.password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AuthResponse(user=user_to_response(user), token=token, expires_at=expires_at)

    @app.post("/auth/login", response_model=AuthResponse)
    def login(request: AuthRequest) -> AuthResponse:
        config = load_config()
        try:
            user, token, expires_at = login_user(
                config,
                username=request.username,
                password=request.password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return AuthResponse(user=user_to_response(user), token=token, expires_at=expires_at)

    @app.post("/auth/logout")
    def logout(authorization: str | None = Header(default=None)) -> dict[str, str]:
        config = load_config()
        token = bearer_token(authorization)
        if token:
            logout_user(config, token=token)
        return {"status": "ok"}

    @app.post("/auth/token/refresh", response_model=AuthResponse)
    def refresh_token(authorization: str | None = Header(default=None)) -> AuthResponse:
        config = load_config()
        token = bearer_token(authorization)
        if not token:
            raise HTTPException(status_code=401, detail="请先登录")
        try:
            user, next_token, expires_at = refresh_session_token(config, current_token=token)
        except ValueError as exc:
            status_code = 401 if str(exc) == "请先登录" else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return AuthResponse(user=user_to_response(user), token=next_token, expires_at=expires_at)

    @app.get("/auth/me", response_model=UserResponse)
    def me(authorization: str | None = Header(default=None)) -> UserResponse:
        user = require_current_user(authorization=authorization)
        return user_to_response(user)

    @app.patch("/auth/me", response_model=UserResponse)
    def update_me(
        request: ProfileUpdateRequest,
        authorization: str | None = Header(default=None),
    ) -> UserResponse:
        config = load_config()
        current = require_current_user(authorization=authorization)
        try:
            user = update_user_profile(
                config,
                user_id=current.id,
                username=request.username,
                display_name=request.display_name,
                avatar_url=request.avatar_url,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return user_to_response(user)

    @app.patch("/auth/password")
    def change_password(
        request: PasswordChangeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        config = load_config()
        current = require_current_user(authorization=authorization)
        try:
            change_user_password(
                config,
                user_id=current.id,
                current_password=request.current_password,
                new_password=request.new_password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.get("/admin/users", response_model=UserListResponse)
    def admin_users(
        authorization: str | None = Header(default=None),
        q: str = Query(default="", max_length=80),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> UserListResponse:
        config = load_config()
        require_admin(config=config, authorization=authorization)
        users = list_public_users(config, query=q, limit=limit, offset=offset)
        total = count_public_users(config, query=q)
        return UserListResponse(
            users=[user_to_response(user) for user in users],
            total=total,
            limit=limit,
            offset=offset,
            query=q,
        )

    @app.patch("/admin/users/{user_id}/status", response_model=UserResponse)
    def admin_update_user_status(
        user_id: str,
        request: UserStatusRequest,
        authorization: str | None = Header(default=None),
    ) -> UserResponse:
        config = load_config()
        actor = require_admin(config=config, authorization=authorization)
        try:
            user = set_user_status(config, actor_id=actor.id, user_id=user_id, status=request.status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return user_to_response(user)

    @app.patch("/admin/users/bulk", response_model=UserListResponse)
    def admin_bulk_update_users(
        request: AdminUserBulkUpdateRequest,
        authorization: str | None = Header(default=None),
    ) -> UserListResponse:
        config = load_config()
        actor = require_admin(config=config, authorization=authorization)
        try:
            users = bulk_update_users(
                config,
                actor_id=actor.id,
                updates=[item.model_dump(exclude_unset=True) for item in request.users],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UserListResponse(
            users=[user_to_response(user) for user in users],
            total=len(users),
            limit=len(users),
            offset=0,
            query="",
        )

    @app.get("/admin/settings", response_model=AdminSettingsResponse)
    def admin_settings(authorization: str | None = Header(default=None)) -> AdminSettingsResponse:
        config = load_config()
        require_admin(config=config, authorization=authorization)
        return admin_settings_response(config)

    @app.patch("/admin/settings/registration", response_model=AdminSettingsResponse)
    def admin_update_registration_settings(
        request: RegistrationSettingsRequest,
        authorization: str | None = Header(default=None),
    ) -> AdminSettingsResponse:
        config = load_config()
        require_admin(config=config, authorization=authorization)
        set_registration_enabled(config, enabled=request.registration_enabled)
        return admin_settings_response(config)

    @app.post("/admin/announcements", response_model=AnnouncementResponse)
    def admin_create_announcement(
        request: AnnouncementRequest,
        authorization: str | None = Header(default=None),
    ) -> AnnouncementResponse:
        config = load_config()
        user = require_admin(config=config, authorization=authorization)
        try:
            row = create_announcement(
                config,
                title=request.title,
                content=request.content,
                author_id=user.id,
                link_url=request.link_url,
                link_label=request.link_label,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AnnouncementResponse(**row, author_name=user.display_name)

    @app.delete("/admin/announcements/{announcement_id}")
    def admin_delete_announcement(
        announcement_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        config = load_config()
        require_admin(config=config, authorization=authorization)
        removed = delete_announcement(config, announcement_id=announcement_id)
        return {"status": "deleted" if removed else "not_found", "announcement_id": announcement_id}

    @app.get("/announcements", response_model=AnnouncementListResponse)
    def public_announcements(limit: int = 5) -> AnnouncementListResponse:
        config = load_config()
        return AnnouncementListResponse(
            announcements=[AnnouncementResponse(**row) for row in list_announcements(config, limit=limit)]
        )

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
            pending_source = create_source_task(
                config=config,
                tenant_id=auth_context.tenant_id,
                path=saved_path,
                acl_groups=auth_context.acl_groups or body_acl_groups or ["engineering"],
                doc_version=doc_version,
            )
            accepted = submit_upload_ingestion_job(
                pending_source=pending_source,
                saved_path=saved_path,
                tenant_id=auth_context.tenant_id,
                acl_groups=auth_context.acl_groups or body_acl_groups or ["engineering"],
                doc_version=doc_version,
                language=language,
            )
            if not accepted:
                fail_source_task(
                    config=config,
                    tenant_id=auth_context.tenant_id,
                    source=pending_source,
                    error="Ingestion queue is full. Please retry later.",
                )
                raise HTTPException(status_code=503, detail="Ingestion queue is full. Please retry later.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SourceUploadResponse(
            status="queued",
            sources=[source_to_response(pending_source)],
            document_count=0,
            chunk_count=0,
        )

    @app.get("/sources/content/{doc_id:path}", response_model=SourceContentResponse)
    def source_content(
        doc_id: str,
        tenant_id: str = "team_a",
        doc_version: int | None = None,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> SourceContentResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        content = get_source_content(
            config=config,
            tenant_id=auth_context.tenant_id,
            doc_id=doc_id,
            doc_version=doc_version,
        )
        if content is None:
            raise HTTPException(status_code=404, detail="Source not found")
        return SourceContentResponse(**content.__dict__)

    @app.get("/source-assets/{asset_path:path}")
    def source_asset(
        asset_path: str,
        tenant_id: str = "team_a",
    ) -> Response:
        config = load_config()
        if asset_path.startswith("__s3__/"):
            object_uri = unquote_object_uri(asset_path[len("__s3__/") :])
            try:
                body = read_object_bytes_by_uri(object_uri)
            except Exception:
                raise HTTPException(status_code=404, detail="Asset not found") from None
            media_type = mimetypes.guess_type(object_uri)[0] or "application/octet-stream"
            if not media_type.startswith("image/"):
                raise HTTPException(status_code=404, detail="Asset not found")
            return Response(content=body, media_type=media_type)
        asset_parts = asset_path.split("/")
        if len(asset_parts) < 3 or asset_parts[0] != "uploads":
            raise HTTPException(status_code=404, detail="Asset not found")
        requested_tenant = asset_parts[1]
        if requested_tenant != tenant_id:
            raise HTTPException(status_code=404, detail="Asset not found")
        try:
            object_store_dir = config.object_store_dir.expanduser().resolve()
            path = (object_store_dir / asset_path).expanduser().resolve()
            path.relative_to(object_store_dir)
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="Asset not found") from None
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if not media_type.startswith("image/"):
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(path, media_type=media_type)

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

    @app.patch("/sources/{doc_id:path}", response_model=RenameSourceResponse)
    def rename_source_endpoint(
        doc_id: str,
        request: RenameSourceRequest,
        tenant_id: str = "team_a",
        doc_version: int | None = None,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> RenameSourceResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        try:
            source = rename_source(
                config=config,
                tenant_id=auth_context.tenant_id,
                doc_id=doc_id,
                doc_version=doc_version,
                title=request.title,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RenameSourceResponse(status="renamed", doc_id=source.doc_id, title=source.title)

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
            hits=[hit_to_response(hit, config=config, tenant_id=auth_context.tenant_id) for hit in result.hits],
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
            citations=[hit_to_response(hit, config=config, tenant_id=auth_context.tenant_id) for hit in result.hits],
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

    @app.post("/query/stream")
    def query_stream(
        request: QueryRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> StreamingResponse:
        config = load_config()
        auth_context = resolve_auth_context(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            request=request,
        )

        def stream_events():
            event_queue: Queue[dict[str, object] | None] = Queue()

            def emit(event_type: str, payload: dict[str, object]) -> None:
                event_queue.put({"type": event_type, **payload})

            def emit_stage_event(payload: dict[str, object]) -> None:
                emit("stage", payload)

            def run_query() -> None:
                try:
                    emit(
                        "stage",
                        {
                            "stage": "start",
                            "status": "done",
                            "label": "接收问题",
                            "detail": "已收到问题，正在准备 RAG 调用链。",
                        },
                    )
                    result = resolve_answer_result(request, auth_context, stage_callback=emit_stage_event)
                    response = QueryResponse(
                        request_id=result.request_id,
                        answer=result.answer,
                        citations=[
                            hit_to_response(hit, config=config, tenant_id=auth_context.tenant_id)
                            for hit in result.hits
                        ],
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
                    emit("result", response.model_dump())
                except Exception as exc:  # noqa: BLE001 - streamed API must serialize failures.
                    emit("error", {"detail": str(exc) or exc.__class__.__name__})
                finally:
                    event_queue.put(None)

            threading.Thread(target=run_query, daemon=True).start()
            while True:
                event = event_queue.get()
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(stream_events(), media_type="application/x-ndjson")

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(
        request: FeedbackRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> FeedbackResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=request.tenant_id,
            acl_groups=request.acl_groups,
        )
        append_event(
            config.runtime_dir,
            "feedback_events",
            {
                **request.model_dump(),
                "tenant_id": auth_context.tenant_id,
                "acl_groups": auth_context.acl_groups,
                "auth_context": auth_context.summary(),
            },
        )
        return FeedbackResponse(
            status="accepted",
            request_id=request.request_id,
        )

    @app.get("/conversations", response_model=ConversationListResponse)
    def conversations(
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> ConversationListResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        return ConversationListResponse(
            conversations=[
                conversation_to_list_item(conversation)
                for conversation in list_conversations(config, tenant_id=auth_context.tenant_id)
            ]
        )

    @app.post("/conversations", response_model=ConversationResponse)
    def upsert_conversation(
        request: ConversationUpsertRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> ConversationResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=request.tenant_id,
            acl_groups=[],
        )
        conversation = save_conversation(
            config,
            tenant_id=auth_context.tenant_id,
            conversation_id=request.id,
            title=request.title,
            messages=[message_request_to_domain(message) for message in request.messages],
            source_doc_ids=request.source_doc_ids,
        )
        return conversation_to_response(conversation)

    @app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
    def get_conversation(
        conversation_id: str,
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> ConversationResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        conversation = load_conversation(
            config,
            tenant_id=auth_context.tenant_id,
            conversation_id=conversation_id,
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation_to_response(conversation)

    @app.delete("/conversations/{conversation_id}", response_model=DeleteConversationResponse)
    def remove_conversation(
        conversation_id: str,
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> DeleteConversationResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        removed = delete_conversation(
            config,
            tenant_id=auth_context.tenant_id,
            conversation_id=conversation_id,
        )
        return DeleteConversationResponse(
            status="deleted" if removed else "not_found",
            conversation_id=conversation_id,
        )

    @app.get("/artifacts", response_model=ArtifactListResponse)
    def artifacts(
        tenant_id: str = "team_a",
        workspace_id: str = "",
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
        migrate_legacy_artifacts(config, tenant_id=auth_context.tenant_id)
        return ArtifactListResponse(
            artifacts=[
                artifact_to_response(artifact)
                for artifact in list_metadata_artifacts(
                    config,
                    tenant_id=auth_context.tenant_id,
                    workspace_id=workspace_id,
                )
            ]
        )

    @app.post("/artifacts/mindmap", response_model=MindMapArtifactResponse)
    def create_mindmap(
        request: MindMapRequest,
        background_tasks: BackgroundTasks,
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
        artifact = pending_artifact(
            title=request.title,
            tenant_id=auth_context.tenant_id,
            workspace_id=request.workspace_id,
            source_doc_ids=request.source_doc_ids,
            artifact_type="mindmap",
        )
        save_metadata_artifact(config, artifact)
        background_tasks.add_task(
            build_mindmap_background,
            artifact,
            request.context_limit,
        )
        return artifact_to_response(artifact)

    @app.post("/artifacts/table", response_model=MindMapArtifactResponse)
    def create_table(
        request: MindMapRequest,
        background_tasks: BackgroundTasks,
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
        artifact = pending_artifact(
            title=request.title,
            tenant_id=auth_context.tenant_id,
            workspace_id=request.workspace_id,
            source_doc_ids=request.source_doc_ids,
            artifact_type="table",
        )
        save_metadata_artifact(config, artifact)
        background_tasks.add_task(build_table_background, artifact)
        return artifact_to_response(artifact)

    @app.get("/artifacts/{artifact_id}", response_model=MindMapArtifactResponse)
    def get_artifact(
        artifact_id: str,
        tenant_id: str = "team_a",
        workspace_id: str = "",
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
        artifact = load_metadata_artifact(
            config,
            tenant_id=auth_context.tenant_id,
            artifact_id=artifact_id,
            workspace_id=workspace_id,
        )
        if artifact is None:
            artifact = None if workspace_id else load_artifact(config, tenant_id=auth_context.tenant_id, artifact_id=artifact_id)
            if artifact is not None:
                save_metadata_artifact(config, artifact)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return artifact_to_response(artifact)

    @app.delete("/artifacts/{artifact_id}", response_model=DeleteArtifactResponse)
    def remove_artifact(
        artifact_id: str,
        tenant_id: str = "team_a",
        workspace_id: str = "",
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
        removed = delete_metadata_artifact(
            config,
            tenant_id=auth_context.tenant_id,
            artifact_id=artifact_id,
            workspace_id=workspace_id,
        )
        legacy_removed = False if workspace_id else delete_artifact(config, tenant_id=auth_context.tenant_id, artifact_id=artifact_id)
        return DeleteArtifactResponse(
            status="deleted" if removed or legacy_removed else "not_found",
            artifact_id=artifact_id,
        )

    @app.patch("/artifacts/{artifact_id}", response_model=RenameArtifactResponse)
    def rename_artifact(
        artifact_id: str,
        request: RenameArtifactRequest,
        tenant_id: str = "team_a",
        workspace_id: str = "",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> RenameArtifactResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        artifact = load_metadata_artifact(
            config,
            tenant_id=auth_context.tenant_id,
            artifact_id=artifact_id,
            workspace_id=workspace_id,
        )
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        updated_artifact = replace(
            artifact,
            title=request.title,
            updated_at=int(time.time() * 1000)
        )
        save_metadata_artifact(config, updated_artifact)

        return RenameArtifactResponse(
            status="renamed",
            artifact_id=artifact_id,
            title=updated_artifact.title,
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
        user = authenticate_token(config, token=bearer_token(authorization))
        if user is not None:
            return build_auth_context(
                config=config,
                header_tenant_id=user.tenant_id,
                header_acl_groups="engineering",
                body_tenant_id=request.tenant_id,
                body_acl_groups=request.acl_groups,
            )
        if not config.api_token:
            raise ValueError("请先登录")
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


def pending_artifact(
    *,
    title: str,
    tenant_id: str,
    workspace_id: str,
    source_doc_ids: list[str],
    artifact_type: str,
) -> MindMapArtifact:
    timestamp = int(time.time() * 1000)
    prefix = "table" if artifact_type == "table" else "mindmap"
    return MindMapArtifact(
        id=f"{prefix}-{uuid.uuid4().hex[:12]}",
        title=title,
        status="generating",
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_doc_ids=source_doc_ids,
        created_at=timestamp,
        updated_at=timestamp,
        artifact_type=artifact_type,
        root=None,
        table=None,
    )


def build_mindmap_background(artifact: MindMapArtifact, context_limit: int) -> None:
    config = load_config()
    try:
        root = build_mindmap_root(
            title=artifact.title,
            config=config,
            tenant_id=artifact.tenant_id,
            source_doc_ids=artifact.source_doc_ids,
            batch_chunk_count=context_limit,
        )
        save_metadata_artifact(
            config,
            replace(artifact, status="ready", root=root, updated_at=int(time.time() * 1000)),
        )
    except Exception as exc:
        fail_metadata_artifact(config, artifact, str(exc))


def build_table_background(artifact: MindMapArtifact) -> None:
    config = load_config()
    try:
        table = build_llm_table(
            title=artifact.title,
            config=config,
            tenant_id=artifact.tenant_id,
            source_doc_ids=artifact.source_doc_ids,
        )
        save_metadata_artifact(
            config,
            replace(artifact, status="ready", table=table, updated_at=int(time.time() * 1000)),
        )
    except Exception as exc:
        fail_metadata_artifact(config, artifact, str(exc))


def migrate_legacy_artifacts(config, *, tenant_id: str) -> None:
    for artifact in list_artifacts(config, tenant_id=tenant_id):
        if load_metadata_artifact(config, tenant_id=tenant_id, artifact_id=artifact.id) is None:
            save_metadata_artifact(config, artifact)


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
        user = authenticate_token(config, token=bearer_token(authorization))
        if user is not None:
            return build_auth_context(
                config=config,
                header_tenant_id=user.tenant_id,
                header_acl_groups="engineering",
                body_tenant_id=tenant_id,
                body_acl_groups=acl_groups,
            )
        if not config.api_token:
            raise ValueError("请先登录")
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


def materialize_query_image(request: QueryRequest) -> str | None:
    if not request.image_data_url:
        return None
    prefix, separator, encoded = request.image_data_url.partition(",")
    if separator != "," or not prefix.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="image_data_url must be a data:image URL")
    media_type = prefix.removeprefix("data:").split(";", 1)[0]
    extension = media_type.split("/", 1)[1].lower()
    if extension == "jpeg":
        extension = "jpg"
    if extension not in {"png", "jpg", "webp", "gif"}:
        raise HTTPException(status_code=400, detail="Unsupported query image type")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid query image data") from exc
    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Query image is too large")
    config = load_config()
    query_dir = config.runtime_dir / "query_images"
    query_dir.mkdir(parents=True, exist_ok=True)
    image_path = query_dir / f"{uuid.uuid4().hex}.{extension}"
    image_path.write_bytes(image_bytes)
    return str(image_path)


def resolve_search_result(request: SearchRequest, auth_context):
    if request.query_mode == "multimodal":
        image_query_path = materialize_query_image(request)
        return retrieve_multimodal(
            request.query,
            text_query=request.query,
            image_query_path=image_query_path,
            tenant_id=auth_context.tenant_id,
            candidate_limit=request.candidate_limit,
            context_limit=request.context_limit,
            acl_groups=auth_context.acl_groups or None,
            doc_version=request.doc_version,
            doc_ids=request.doc_ids or None,
            source_types=request.source_types or None,
            include_all_sources=request.include_all_sources,
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
        include_all_sources=request.include_all_sources,
        history=request.history,
        request_id=request.request_id,
    )


def resolve_answer_result(request: QueryRequest, auth_context, stage_callback=None):
    if request.query_mode == "multimodal":
        image_query_path = materialize_query_image(request)
        return answer_multimodal_query(
            request.query,
            text_query=request.query,
            image_query_path=image_query_path,
            tenant_id=auth_context.tenant_id,
            candidate_limit=request.candidate_limit,
            context_limit=request.context_limit,
            acl_groups=auth_context.acl_groups or None,
            doc_version=request.doc_version,
            doc_ids=request.doc_ids or None,
            source_types=request.source_types or None,
            include_all_sources=request.include_all_sources,
            history=request.history,
            request_id=request.request_id,
            answer_query=request.query,
            stage_callback=stage_callback,
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
        include_all_sources=request.include_all_sources,
        history=request.history,
        request_id=request.request_id,
        stage_callback=stage_callback,
    )


def hit_to_response(hit, *, config=None, tenant_id: str = "") -> HitResponse:
    metadata = hit.metadata
    if config is not None and tenant_id:
        metadata = resolve_metadata_display_block_urls(config=config, tenant_id=tenant_id, metadata=metadata)
    return HitResponse(
        doc_id=hit.doc_id,
        title=hit.title,
        source_uri=hit.source_uri,
        source_type=hit.source_type,
        chunk_index=hit.chunk_index,
        score=hit.score,
        rerank_score=hit.rerank_score,
        acl_groups=hit.acl_groups,
        metadata=metadata,
        text=hit.text,
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
        child_doc_ids=source.child_doc_ids,
        error=getattr(source, "error", ""),
    )


def artifact_to_response(artifact) -> MindMapArtifactResponse:
    return MindMapArtifactResponse(
        id=artifact.id,
        title=artifact.title,
        status=artifact.status,
        tenant_id=artifact.tenant_id,
        workspace_id=artifact.workspace_id,
        source_doc_ids=artifact.source_doc_ids,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
        artifact_type=artifact.artifact_type,
        root=artifact.root,
        table=artifact.table,
        error=artifact.error,
    )


def message_request_to_domain(message: ConversationMessageRequest) -> ConversationMessage:
    return ConversationMessage(
        id=message.id,
        role=message.role,  # type: ignore[arg-type]
        content=message.content,
        status=message.status,  # type: ignore[arg-type]
        request_id=message.request_id,
        citations=[citation.model_dump() for citation in message.citations],
        image_data_url=message.image_data_url,
        created_at=message.created_at,
        feedback_rating=message.feedback_rating,
        rag_progress=message.rag_progress,
    )


def conversation_to_response(conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        tenant_id=conversation.tenant_id,
        title=conversation.title,
        messages=[
            ConversationMessageRequest(
                id=message.id,
                role=message.role,
                content=message.content,
                status=message.status,
                request_id=message.request_id,
                citations=[HitResponse(**citation) for citation in message.citations],
                image_data_url=message.image_data_url,
                created_at=message.created_at,
                feedback_rating=message.feedback_rating,
                rag_progress=message.rag_progress,
            )
            for message in conversation.messages
        ],
        source_doc_ids=conversation.source_doc_ids,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def conversation_to_list_item(conversation) -> ConversationListItemResponse:
    return ConversationListItemResponse(
        id=conversation.id,
        tenant_id=conversation.tenant_id,
        title=conversation.title,
        message_count=len(conversation.messages),
        source_doc_ids=conversation.source_doc_ids,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def user_to_response(user) -> UserResponse:
    return UserResponse(**user.public_dict())


def admin_settings_response(config) -> AdminSettingsResponse:
    latest = list_announcements(config, limit=1)
    return AdminSettingsResponse(
        registration_enabled=is_registration_enabled(config),
        latest_announcement=AnnouncementResponse(**latest[0]) if latest else None,
    )


def require_current_user(*, authorization: str | None):
    from fastapi import HTTPException

    config = load_config()
    user = authenticate_token(config, token=bearer_token(authorization))
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def require_admin(*, config, authorization: str | None):
    from fastapi import HTTPException

    user = authenticate_token(config, token=bearer_token(authorization))
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="127.0.0.1", port=8008, reload=False)
