from __future__ import annotations

from dataclasses import dataclass

from rag_core.config import RagConfig


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    acl_groups: list[str]
    source: str
    user_id: str = ""
    username: str = ""

    def summary(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "acl_groups": self.acl_groups,
            "source": self.source,
            "user_id": self.user_id,
            "username": self.username,
        }


def parse_acl_groups(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_auth_context(
    *,
    config: RagConfig,
    header_tenant_id: str | None,
    header_acl_groups: str | None,
    body_tenant_id: str,
    body_acl_groups: list[str],
    user_id: str = "",
    username: str = "",
) -> AuthContext:
    if header_tenant_id:
        return AuthContext(
            tenant_id=header_tenant_id,
            acl_groups=parse_acl_groups(header_acl_groups),
            source="headers",
            user_id=user_id,
            username=username,
        )

    if config.require_auth_context:
        raise ValueError(
            "Missing auth context headers: X-RAG-Tenant-ID and X-RAG-ACL-Groups"
        )

    return AuthContext(
        tenant_id=body_tenant_id,
        acl_groups=body_acl_groups,
        source="request_body_compat",
        user_id=user_id,
        username=username,
    )


def validate_bearer_token(
    *,
    config: RagConfig,
    authorization: str | None,
) -> None:
    if not config.api_token:
        return
    expected = f"Bearer {config.api_token}"
    if authorization != expected:
        raise ValueError("Invalid or missing bearer token")
