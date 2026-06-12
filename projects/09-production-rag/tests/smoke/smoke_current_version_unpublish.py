from __future__ import annotations

import tempfile
from pathlib import Path

from rag_core.types import SourceDocument
from rag_core.versioning import (
    load_all_current_versions,
    publish_current_versions,
    unpublish_current_version,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        object_store_dir = Path(tmp) / "object_store"
        docs = [
            SourceDocument(
                tenant_id="team_a",
                doc_id="versioned-runbook",
                doc_version=1,
                source_type="md",
                source_uri="memory://versioned-runbook-v1",
                title="Versioned Runbook v1",
                text="历史版本。",
                acl_groups=["ops"],
            ),
            SourceDocument(
                tenant_id="team_a",
                doc_id="versioned-runbook",
                doc_version=2,
                source_type="md",
                source_uri="memory://versioned-runbook-v2",
                title="Versioned Runbook v2",
                text="当前版本。",
                acl_groups=["ops"],
            ),
            SourceDocument(
                tenant_id="team_b",
                doc_id="versioned-runbook",
                doc_version=3,
                source_type="md",
                source_uri="memory://team-b-versioned-runbook-v3",
                title="Team B Runbook",
                text="其他租户当前版本。",
                acl_groups=["ops"],
            ),
        ]
        publish_current_versions(object_store_dir, docs)
        assert load_all_current_versions(object_store_dir) == {
            "team_a": {"versioned-runbook": 2},
            "team_b": {"versioned-runbook": 3},
        }

        assert not unpublish_current_version(
            object_store_dir,
            tenant_id="team_a",
            doc_id="versioned-runbook",
            doc_version=1,
        )
        assert load_all_current_versions(object_store_dir)["team_a"] == {
            "versioned-runbook": 2
        }

        assert unpublish_current_version(
            object_store_dir,
            tenant_id="team_a",
            doc_id="versioned-runbook",
            doc_version=2,
        )
        assert load_all_current_versions(object_store_dir) == {
            "team_b": {"versioned-runbook": 3},
        }

        assert unpublish_current_version(
            object_store_dir,
            tenant_id="team_b",
            doc_id="versioned-runbook",
        )
        assert load_all_current_versions(object_store_dir) == {}

    print("smoke_current_version_unpublish=ok")


if __name__ == "__main__":
    main()
