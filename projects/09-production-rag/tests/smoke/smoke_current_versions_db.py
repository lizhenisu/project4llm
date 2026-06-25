from __future__ import annotations

from types import SimpleNamespace
import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.database import connect_metadata_db
from rag_core.types import SourceDocument
from rag_core.versioning import (
    CURRENT_VERSIONS_PATH,
    load_current_versions,
    publish_current_versions,
    unpublish_current_version,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = SimpleNamespace(
            runtime_dir=root / "runtime",
            object_store_dir=root / "object_store",
        )
        docs = [
            SourceDocument(
                tenant_id="team_a",
                doc_id="doc-a",
                doc_version=1,
                source_type="md",
                source_uri="memory://doc-a-v1",
                title="Doc A v1",
                text="old",
            ),
            SourceDocument(
                tenant_id="team_a",
                doc_id="doc-a",
                doc_version=2,
                source_type="md",
                source_uri="memory://doc-a-v2",
                title="Doc A v2",
                text="new",
            ),
            SourceDocument(
                tenant_id="team_a",
                doc_id="doc-b",
                doc_version=1,
                source_type="md",
                source_uri="memory://doc-b-v1",
                title="Doc B",
                text="current",
            ),
        ]
        current = publish_current_versions(config.object_store_dir, docs, config=config)
        assert current == {"doc-a": 2, "doc-b": 1}
        assert not (config.object_store_dir / CURRENT_VERSIONS_PATH).exists()
        assert load_current_versions(config.object_store_dir, tenant_id="team_a", config=config) == current

        assert not unpublish_current_version(
            config.object_store_dir,
            tenant_id="team_a",
            doc_id="doc-a",
            doc_version=1,
            config=config,
        )
        assert unpublish_current_version(
            config.object_store_dir,
            tenant_id="team_a",
            doc_id="doc-a",
            doc_version=2,
            config=config,
        )
        assert load_current_versions(config.object_store_dir, tenant_id="team_a", config=config) == {
            "doc-b": 1,
        }

        legacy_store = root / "legacy_object_store"
        legacy_config = SimpleNamespace(
            runtime_dir=root / "legacy_runtime",
            object_store_dir=legacy_store,
        )
        legacy_store.mkdir(parents=True, exist_ok=True)
        (legacy_store / CURRENT_VERSIONS_PATH).write_text(
            '{"team_legacy":{"legacy-doc":7}}',
            encoding="utf-8",
        )
        assert load_current_versions(
            legacy_config.object_store_dir,
            tenant_id="team_legacy",
            config=legacy_config,
        ) == {"legacy-doc": 7}
        with connect_metadata_db(legacy_config) as conn:
            row = conn.execute(
                """
                SELECT doc_version
                FROM current_source_versions
                WHERE tenant_id = ? AND doc_id = ?
                """,
                ("team_legacy", "legacy-doc"),
            ).fetchone()
        assert row is not None and int(row["doc_version"]) == 7
        assert unpublish_current_version(
            legacy_config.object_store_dir,
            tenant_id="team_legacy",
            doc_id="legacy-doc",
            doc_version=7,
            config=legacy_config,
        )
        assert load_current_versions(
            legacy_config.object_store_dir,
            tenant_id="team_legacy",
            config=legacy_config,
        ) == {}

    print("smoke_current_versions_db=ok")


if __name__ == "__main__":
    main()
