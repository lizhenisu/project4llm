from __future__ import annotations

import sqlite3
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from rag_core.config import RagConfig


SCHEMA_VERSION = 1


def metadata_db_path(config: RagConfig) -> Path:
    db_dir = config.runtime_dir / "db"
    path = db_dir / "metadata.db"
    legacy_path = config.runtime_dir / "metadata.db"
    if legacy_path.exists() and not path.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_path, path)
    return path


@contextmanager
def connect_metadata_db(config: RagConfig) -> Iterator[sqlite3.Connection]:
    path = metadata_db_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        initialize_schema(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
            tenant_id TEXT NOT NULL UNIQUE,
            avatar_url TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            created_at INTEGER NOT NULL,
            last_login_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

        CREATE TABLE IF NOT EXISTS announcements (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            author_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(author_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_announcements_created_at ON announcements(created_at DESC);

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            title TEXT NOT NULL,
            source_doc_ids TEXT NOT NULL DEFAULT '[]',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_tenant_updated ON conversations(tenant_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL,
            request_id TEXT,
            citations TEXT NOT NULL DEFAULT '[]',
            image_data_url TEXT,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id, created_at);

        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            artifact_type TEXT NOT NULL DEFAULT 'mindmap',
            source_doc_ids TEXT NOT NULL DEFAULT '[]',
            root TEXT,
            table_json TEXT,
            error TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_artifacts_tenant_updated ON artifacts(tenant_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS source_tasks (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            title TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_uri TEXT NOT NULL,
            doc_version INTEGER NOT NULL,
            acl_groups TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_source_tasks_tenant_updated ON source_tasks(tenant_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS source_title_overrides (
            tenant_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            doc_version INTEGER NOT NULL,
            title TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY(tenant_id, doc_id, doc_version)
        );
        CREATE INDEX IF NOT EXISTS idx_source_title_overrides_tenant ON source_title_overrides(tenant_id);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
        (str(SCHEMA_VERSION),),
    )
    ensure_column(conn, table="messages", column="request_id", definition="TEXT")
    ensure_column(conn, table="messages", column="feedback_rating", definition="INTEGER")
    ensure_column(conn, table="messages", column="image_data_url", definition="TEXT")
    ensure_column(conn, table="users", column="avatar_url", definition="TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, table="users", column="status", definition="TEXT NOT NULL DEFAULT 'active'")


def ensure_column(conn: sqlite3.Connection, *, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
