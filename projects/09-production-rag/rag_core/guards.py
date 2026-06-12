from __future__ import annotations

import re


TENANT_PATTERN = re.compile(r"\bteam_[a-zA-Z0-9]+\b")


def mentions_other_tenant(query: str, allowed_tenant_id: str) -> bool:
    mentions = {match.group(0) for match in TENANT_PATTERN.finditer(query)}
    return any(tenant != allowed_tenant_id for tenant in mentions)

