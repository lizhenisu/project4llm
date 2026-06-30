from __future__ import annotations

import re
import os
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from rag_core.config import RagConfig


SCHEMA_VERSION = 1
_POSTGRES_SCHEMA_LOCK = threading.Lock()
_POSTGRES_SCHEMA_INITIALIZED_URLS: set[str] = set()
_POSTGRES_POOLS_LOCK = threading.Lock()
_POSTGRES_POOLS: dict[tuple[str, int], PostgresConnectionPool] = {}


def metadata_db_path(config: RagConfig) -> Path:
    db_dir = config.runtime_dir / "db"
    path = db_dir / "metadata.db"
    legacy_path = config.runtime_dir / "metadata.db"
    if legacy_path.exists() and not path.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_path, path)
    return path


@contextmanager
def connect_metadata_db(config: RagConfig) -> Iterator[Any]:
    metadata_database_url = getattr(config, "metadata_database_url", None)
    if metadata_database_url:
        with connect_postgres_metadata_db(metadata_database_url) as conn:
            yield conn
        return
    with connect_sqlite_metadata_db(config) as conn:
        yield conn


@contextmanager
def connect_sqlite_metadata_db(config: RagConfig) -> Iterator[sqlite3.Connection]:
    path = metadata_db_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        initialize_sqlite_schema(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def connect_postgres_metadata_db(database_url: str) -> Iterator[PostgresConnection]:
    pool = postgres_connection_pool(database_url)
    raw_conn = pool.acquire()
    conn = PostgresConnection(raw_conn)
    reusable = True
    try:
        ensure_postgres_schema_initialized(database_url, conn)
        yield conn
        raw_conn.commit()
    except Exception:
        try:
            raw_conn.rollback()
        except Exception:
            reusable = False
        raise
    finally:
        pool.release(raw_conn, reusable=reusable)


def postgres_connection_pool(database_url: str) -> PostgresConnectionPool:
    max_size = env_int("RAG_METADATA_POOL_SIZE", 16)
    key = (database_url, max_size)
    with _POSTGRES_POOLS_LOCK:
        pool = _POSTGRES_POOLS.get(key)
        if pool is None:
            pool = PostgresConnectionPool(database_url=database_url, max_size=max_size)
            _POSTGRES_POOLS[key] = pool
        return pool


def metadata_pool_metrics_snapshot() -> dict[str, Any]:
    with _POSTGRES_POOLS_LOCK:
        pools = list(_POSTGRES_POOLS.values())
    return {
        "pools": [pool.snapshot() for pool in pools],
        "pool_count": len(pools),
        "default_pool_size": env_int("RAG_METADATA_POOL_SIZE", 16),
        "acquire_timeout_seconds": env_float("RAG_METADATA_POOL_TIMEOUT_SECONDS", 10.0),
    }


class PostgresConnectionPool:
    def __init__(self, *, database_url: str, max_size: int) -> None:
        self.database_url = database_url
        self.max_size = max(1, int(max_size))
        self._condition = threading.Condition()
        self._idle: list[Any] = []
        self._total = 0
        self._borrowed = 0
        self._waits_total = 0
        self._timeouts_total = 0
        self._created_total = 0
        self._reused_total = 0

    def acquire(self) -> Any:
        timeout_seconds = env_float("RAG_METADATA_POOL_TIMEOUT_SECONDS", 10.0)
        deadline = time.monotonic() + timeout_seconds
        should_create = False
        with self._condition:
            while True:
                if self._idle:
                    raw_conn = self._idle.pop()
                    self._borrowed += 1
                    self._reused_total += 1
                    return raw_conn
                if self._total < self.max_size:
                    self._total += 1
                    self._borrowed += 1
                    should_create = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._timeouts_total += 1
                    raise TimeoutError(
                        "Timed out waiting for PostgreSQL metadata connection; "
                        f"RAG_METADATA_POOL_SIZE={self.max_size}"
                    )
                self._waits_total += 1
                self._condition.wait(timeout=remaining)
        if should_create:
            try:
                raw_conn = open_postgres_raw_connection(self.database_url)
            except Exception:
                with self._condition:
                    self._borrowed = max(0, self._borrowed - 1)
                    self._total = max(0, self._total - 1)
                    self._condition.notify()
                raise
            with self._condition:
                self._created_total += 1
            return raw_conn
        raise RuntimeError("PostgreSQL metadata pool acquisition reached an unreachable state.")

    def release(self, raw_conn: Any, *, reusable: bool) -> None:
        with self._condition:
            self._borrowed = max(0, self._borrowed - 1)
            if reusable and not is_postgres_connection_closed(raw_conn):
                self._idle.append(raw_conn)
            else:
                close_postgres_connection(raw_conn)
                self._total = max(0, self._total - 1)
            self._condition.notify()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "database": redact_database_url(self.database_url),
                "max_size": self.max_size,
                "total": self._total,
                "idle": len(self._idle),
                "borrowed": self._borrowed,
                "waits_total": self._waits_total,
                "timeouts_total": self._timeouts_total,
                "created_total": self._created_total,
                "reused_total": self._reused_total,
            }


def open_postgres_raw_connection(database_url: str) -> Any:
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(database_url, row_factory=dict_row, connect_timeout=10)


def is_postgres_connection_closed(raw_conn: Any) -> bool:
    return bool(getattr(raw_conn, "closed", False))


def close_postgres_connection(raw_conn: Any) -> None:
    try:
        raw_conn.close()
    except Exception:
        pass


def redact_database_url(database_url: str) -> str:
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", database_url)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        return default


class PostgresConnection:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple | list = ()) -> Any:
        cursor = self._conn.execute(translate_sqlite_sql(sql), params)
        return PostgresCursor(cursor)

    def executescript(self, script: str) -> None:
        for statement in split_sql_script(script):
            self.execute(statement)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


def translate_sqlite_sql(sql: str) -> str:
    stripped = " ".join(sql.strip().split())
    if stripped.startswith("INSERT OR REPLACE INTO schema_meta"):
        converted = re.sub(r"\bINSERT\s+OR\s+REPLACE\b", "INSERT", sql, count=1, flags=re.IGNORECASE)
        converted = converted.rstrip().rstrip(";")
        converted += " ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        return replace_qmark_placeholders(converted)
    if stripped.startswith("INSERT OR REPLACE INTO sessions"):
        converted = re.sub(r"\bINSERT\s+OR\s+REPLACE\b", "INSERT", sql, count=1, flags=re.IGNORECASE)
        converted = converted.rstrip().rstrip(";")
        converted += (
            " ON CONFLICT(token) DO UPDATE SET "
            "user_id = excluded.user_id, "
            "expires_at = excluded.expires_at, "
            "created_at = excluded.created_at"
        )
        return replace_qmark_placeholders(converted)
    return replace_qmark_placeholders(sql)


class CompatRow(dict):
    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class PostgresCursor:
    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return int(self._cursor.rowcount)

    def fetchone(self) -> CompatRow | None:
        row = self._cursor.fetchone()
        return compat_row(row)

    def fetchall(self) -> list[CompatRow]:
        return [item for item in (compat_row(row) for row in self._cursor.fetchall()) if item is not None]


def compat_row(row: Any) -> CompatRow | None:
    if row is None:
        return None
    return CompatRow(row)


def replace_qmark_placeholders(sql: str) -> str:
    parts = sql.split("?")
    if len(parts) == 1:
        return sql
    return "%s".join(parts)


def split_sql_script(script: str) -> list[str]:
    return [statement.strip() for statement in script.split(";") if statement.strip()]


def initialize_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
        (str(SCHEMA_VERSION),),
    )
    ensure_sqlite_columns(conn)


def initialize_postgres_schema(conn: PostgresConnection) -> None:
    conn.executescript(postgres_schema_sql())
    conn.execute(
        "INSERT INTO schema_meta(key, value) VALUES ('version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(SCHEMA_VERSION),),
    )
    ensure_postgres_columns(conn)


def ensure_postgres_schema_initialized(database_url: str, conn: PostgresConnection) -> None:
    if database_url in _POSTGRES_SCHEMA_INITIALIZED_URLS:
        return
    with _POSTGRES_SCHEMA_LOCK:
        if database_url in _POSTGRES_SCHEMA_INITIALIZED_URLS:
            return
        initialize_postgres_schema(conn)
        conn.commit()
        _POSTGRES_SCHEMA_INITIALIZED_URLS.add(database_url)


SCHEMA_SQL = """
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
    profile_name_edit_allowed INTEGER NOT NULL DEFAULT 1,
    avatar_edit_allowed INTEGER NOT NULL DEFAULT 1,
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
    link_url TEXT NOT NULL DEFAULT '',
    link_label TEXT NOT NULL DEFAULT '',
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
    workspace_id TEXT NOT NULL DEFAULT '',
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
    ingestion_stage TEXT NOT NULL DEFAULT 'queued',
    stage_started_at INTEGER NOT NULL DEFAULT 0,
    progress_percent INTEGER NOT NULL DEFAULT 0,
    progress_detail TEXT NOT NULL DEFAULT '',
    eta_seconds INTEGER,
    lease_owner TEXT NOT NULL DEFAULT '',
    lease_expires_at INTEGER NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    requested_doc_version INTEGER,
    next_attempt_at INTEGER NOT NULL DEFAULT 0,
    dead_lettered_at INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_tasks_tenant_updated ON source_tasks(tenant_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS ingestion_stage_stats (
    source_type TEXT NOT NULL,
    stage TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    total_duration_ms INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY(source_type, stage)
);

CREATE TABLE IF NOT EXISTS query_admission_slots (
    scope_type TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    slot_index INTEGER NOT NULL,
    lease_owner TEXT NOT NULL,
    lease_expires_at INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY(scope_type, scope_key, slot_index)
);
CREATE INDEX IF NOT EXISTS idx_query_admission_slots_expiry
ON query_admission_slots(lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_query_admission_slots_owner
ON query_admission_slots(lease_owner);

CREATE TABLE IF NOT EXISTS query_result_cache (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    lease_owner TEXT NOT NULL DEFAULT '',
    lease_expires_at INTEGER NOT NULL DEFAULT 0,
    response_json TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    PRIMARY KEY(tenant_id, request_id)
);
CREATE INDEX IF NOT EXISTS idx_query_result_cache_expiry
ON query_result_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_query_result_cache_status_lease
ON query_result_cache(status, lease_expires_at);

CREATE TABLE IF NOT EXISTS upload_admission_slots (
    scope_type TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    slot_index INTEGER NOT NULL,
    reservation_owner TEXT NOT NULL,
    lease_expires_at INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY(scope_type, scope_key, slot_index)
);
CREATE INDEX IF NOT EXISTS idx_upload_admission_slots_expiry
ON upload_admission_slots(lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_upload_admission_slots_owner
ON upload_admission_slots(reservation_owner);

CREATE TABLE IF NOT EXISTS admission_locks (
    name TEXT PRIMARY KEY,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS source_task_resolutions (
    tenant_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    source_doc_id TEXT NOT NULL,
    doc_version INTEGER NOT NULL,
    resolved_at INTEGER NOT NULL,
    PRIMARY KEY(tenant_id, task_id, source_doc_id, doc_version)
);
CREATE INDEX IF NOT EXISTS idx_source_task_resolutions_source
ON source_task_resolutions(tenant_id, source_doc_id, doc_version);

CREATE TABLE IF NOT EXISTS source_catalog (
    tenant_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    doc_version INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    acl_groups TEXT NOT NULL DEFAULT '[]',
    current INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    child_doc_ids TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(tenant_id, doc_id, doc_version)
);
CREATE INDEX IF NOT EXISTS idx_source_catalog_tenant_updated ON source_catalog(tenant_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS source_title_overrides (
    tenant_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    doc_version INTEGER NOT NULL,
    title TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY(tenant_id, doc_id, doc_version)
);
CREATE INDEX IF NOT EXISTS idx_source_title_overrides_tenant ON source_title_overrides(tenant_id);

CREATE TABLE IF NOT EXISTS current_source_versions (
    tenant_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    doc_version INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY(tenant_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_current_source_versions_tenant ON current_source_versions(tenant_id);
"""


def ensure_sqlite_columns(conn: sqlite3.Connection) -> None:
    ensure_sqlite_column(conn, table="messages", column="request_id", definition="TEXT")
    ensure_sqlite_column(conn, table="messages", column="feedback_rating", definition="INTEGER")
    ensure_sqlite_column(conn, table="messages", column="image_data_url", definition="TEXT")
    ensure_sqlite_column(conn, table="messages", column="rag_progress", definition="TEXT NOT NULL DEFAULT '[]'")
    ensure_sqlite_column(conn, table="users", column="avatar_url", definition="TEXT NOT NULL DEFAULT ''")
    ensure_sqlite_column(conn, table="users", column="status", definition="TEXT NOT NULL DEFAULT 'active'")
    ensure_sqlite_column(conn, table="users", column="profile_name_edit_allowed", definition="INTEGER NOT NULL DEFAULT 1")
    ensure_sqlite_column(conn, table="users", column="avatar_edit_allowed", definition="INTEGER NOT NULL DEFAULT 1")
    ensure_sqlite_column(conn, table="announcements", column="link_url", definition="TEXT NOT NULL DEFAULT ''")
    ensure_sqlite_column(conn, table="announcements", column="link_label", definition="TEXT NOT NULL DEFAULT ''")
    ensure_sqlite_column(conn, table="artifacts", column="workspace_id", definition="TEXT NOT NULL DEFAULT ''")
    ensure_sqlite_column(conn, table="source_catalog", column="child_doc_ids", definition="TEXT NOT NULL DEFAULT '[]'")
    ensure_sqlite_column(conn, table="source_tasks", column="lease_owner", definition="TEXT NOT NULL DEFAULT ''")
    ensure_sqlite_column(conn, table="source_tasks", column="lease_expires_at", definition="INTEGER NOT NULL DEFAULT 0")
    ensure_sqlite_column(conn, table="source_tasks", column="attempt_count", definition="INTEGER NOT NULL DEFAULT 0")
    ensure_sqlite_column(conn, table="source_tasks", column="requested_doc_version", definition="INTEGER")
    ensure_sqlite_column(conn, table="source_tasks", column="next_attempt_at", definition="INTEGER NOT NULL DEFAULT 0")
    ensure_sqlite_column(conn, table="source_tasks", column="dead_lettered_at", definition="INTEGER NOT NULL DEFAULT 0")
    ensure_sqlite_column(conn, table="source_tasks", column="ingestion_stage", definition="TEXT NOT NULL DEFAULT 'queued'")
    ensure_sqlite_column(conn, table="source_tasks", column="stage_started_at", definition="INTEGER NOT NULL DEFAULT 0")
    ensure_sqlite_column(conn, table="source_tasks", column="progress_percent", definition="INTEGER NOT NULL DEFAULT 0")
    ensure_sqlite_column(conn, table="source_tasks", column="progress_detail", definition="TEXT NOT NULL DEFAULT ''")
    ensure_sqlite_column(conn, table="source_tasks", column="eta_seconds", definition="INTEGER")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifacts_tenant_workspace_updated "
        "ON artifacts(tenant_id, workspace_id, updated_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_tasks_status_lease "
        "ON source_tasks(status, lease_expires_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_tasks_status_next_attempt "
        "ON source_tasks(status, next_attempt_at)"
    )


def ensure_postgres_columns(conn: PostgresConnection) -> None:
    ensure_postgres_column(conn, table="messages", column="request_id", definition="TEXT")
    ensure_postgres_column(conn, table="messages", column="feedback_rating", definition="BIGINT")
    ensure_postgres_column(conn, table="messages", column="image_data_url", definition="TEXT")
    ensure_postgres_column(conn, table="messages", column="rag_progress", definition="TEXT NOT NULL DEFAULT '[]'")
    ensure_postgres_column(conn, table="users", column="avatar_url", definition="TEXT NOT NULL DEFAULT ''")
    ensure_postgres_column(conn, table="users", column="status", definition="TEXT NOT NULL DEFAULT 'active'")
    ensure_postgres_column(conn, table="users", column="profile_name_edit_allowed", definition="BIGINT NOT NULL DEFAULT 1")
    ensure_postgres_column(conn, table="users", column="avatar_edit_allowed", definition="BIGINT NOT NULL DEFAULT 1")
    ensure_postgres_column(conn, table="announcements", column="link_url", definition="TEXT NOT NULL DEFAULT ''")
    ensure_postgres_column(conn, table="announcements", column="link_label", definition="TEXT NOT NULL DEFAULT ''")
    ensure_postgres_column(conn, table="artifacts", column="workspace_id", definition="TEXT NOT NULL DEFAULT ''")
    ensure_postgres_column(conn, table="source_catalog", column="child_doc_ids", definition="TEXT NOT NULL DEFAULT '[]'")
    ensure_postgres_column(conn, table="source_tasks", column="lease_owner", definition="TEXT NOT NULL DEFAULT ''")
    ensure_postgres_column(conn, table="source_tasks", column="lease_expires_at", definition="BIGINT NOT NULL DEFAULT 0")
    ensure_postgres_column(conn, table="source_tasks", column="attempt_count", definition="BIGINT NOT NULL DEFAULT 0")
    ensure_postgres_column(conn, table="source_tasks", column="requested_doc_version", definition="BIGINT")
    ensure_postgres_column(conn, table="source_tasks", column="next_attempt_at", definition="BIGINT NOT NULL DEFAULT 0")
    ensure_postgres_column(conn, table="source_tasks", column="dead_lettered_at", definition="BIGINT NOT NULL DEFAULT 0")
    ensure_postgres_column(conn, table="source_tasks", column="ingestion_stage", definition="TEXT NOT NULL DEFAULT 'queued'")
    ensure_postgres_column(conn, table="source_tasks", column="stage_started_at", definition="BIGINT NOT NULL DEFAULT 0")
    ensure_postgres_column(conn, table="source_tasks", column="progress_percent", definition="BIGINT NOT NULL DEFAULT 0")
    ensure_postgres_column(conn, table="source_tasks", column="progress_detail", definition="TEXT NOT NULL DEFAULT ''")
    ensure_postgres_column(conn, table="source_tasks", column="eta_seconds", definition="BIGINT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifacts_tenant_workspace_updated "
        "ON artifacts(tenant_id, workspace_id, updated_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_tasks_status_lease "
        "ON source_tasks(status, lease_expires_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_tasks_status_next_attempt "
        "ON source_tasks(status, next_attempt_at)"
    )


def ensure_sqlite_column(conn: sqlite3.Connection, *, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def postgres_schema_sql() -> str:
    return re.sub(r"\bINTEGER\b", "BIGINT", SCHEMA_SQL)


def ensure_postgres_column(conn: PostgresConnection, *, table: str, column: str, definition: str) -> None:
    row = conn.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = ? AND column_name = ?
        """,
        (table, column),
    ).fetchone()
    if row is None:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def ensure_column(conn: Any, *, table: str, column: str, definition: str) -> None:
    if isinstance(conn, PostgresConnection):
        ensure_postgres_column(conn, table=table, column=column, definition=definition)
    else:
        ensure_sqlite_column(conn, table=table, column=column, definition=definition)
