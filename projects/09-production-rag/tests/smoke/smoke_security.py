from __future__ import annotations

from dataclasses import replace

from rag_core.auth import build_auth_context, validate_bearer_token
from rag_core.config import load_config
from rag_core.guards import mentions_other_tenant
from rag_core.pii import apply_pii_policy, detect_pii


def main() -> None:
    assert detect_pii("联系 test@example.com 或 13800138000")
    redacted = apply_pii_policy(
        "联系 test@example.com 或 13800138000",
        policy="redact",
        label="smoke",
    )
    assert "test@example.com" not in redacted
    assert "13800138000" not in redacted

    config = replace(
        load_config(),
        api_token="demo-token",
        require_auth_context=True,
    )
    validate_bearer_token(config=config, authorization="Bearer demo-token")
    auth_context = build_auth_context(
        config=config,
        header_tenant_id="team_a",
        header_acl_groups="finance,engineering",
        body_tenant_id="ignored_body_tenant",
        body_acl_groups=["ignored_body_acl"],
    )
    assert auth_context.tenant_id == "team_a"
    assert auth_context.acl_groups == ["finance", "engineering"]
    assert auth_context.source == "headers"

    assert mentions_other_tenant("team_b 报销规则", allowed_tenant_id="team_a")
    assert not mentions_other_tenant("team_a 报销规则", allowed_tenant_id="team_a")
    print("smoke_security=ok")


if __name__ == "__main__":
    main()
