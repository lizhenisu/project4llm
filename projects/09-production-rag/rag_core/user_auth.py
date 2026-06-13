from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Any

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db


PBKDF2_ITERATIONS = 240_000
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True)
class User:
    id: str
    username: str
    display_name: str
    role: str
    tenant_id: str
    created_at: int
    last_login_at: int | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
        }


def now_ms() -> int:
    return int(time.time() * 1000)


def normalize_username(username: str) -> str:
    value = username.strip().lower()
    if len(value) < 3 or len(value) > 32:
        raise ValueError("用户名长度需要在 3 到 32 个字符之间")
    if not all(ch.isalnum() or ch in {"_", "-", "."} for ch in value):
        raise ValueError("用户名只能包含字母、数字、下划线、短横线或点")
    return value


def validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("密码至少需要 8 个字符")
    if len(password) > 128:
        raise ValueError("密码不能超过 128 个字符")


def hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PBKDF2_ITERATIONS,
    )
    return digest.hex()


def verify_password(password: str, *, salt: str, password_hash: str) -> bool:
    expected = hash_password(password, salt)
    return hmac.compare_digest(expected, password_hash)


def register_user(
    config: RagConfig,
    *,
    username: str,
    password: str,
    display_name: str | None = None,
) -> User:
    username = normalize_username(username)
    validate_password(password)
    timestamp = now_ms()
    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    user_id = f"user-{uuid.uuid4().hex[:12]}"
    tenant_id = f"tenant-{uuid.uuid4().hex[:12]}"
    display = (display_name or username).strip()[:40] or username
    with connect_metadata_db(config) as conn:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        role = "admin" if user_count == 0 else "user"
        try:
            conn.execute(
                """
                INSERT INTO users(
                    id, username, display_name, password_hash, salt,
                    role, tenant_id, created_at, last_login_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (user_id, username, display, password_hash, salt, role, tenant_id, timestamp),
            )
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                raise ValueError("用户名已存在") from exc
            raise
    return User(
        id=user_id,
        username=username,
        display_name=display,
        role=role,
        tenant_id=tenant_id,
        created_at=timestamp,
    )


def login_user(config: RagConfig, *, username: str, password: str) -> tuple[User, str, int]:
    username = normalize_username(username)
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None or not verify_password(
            password,
            salt=str(row["salt"]),
            password_hash=str(row["password_hash"]),
        ):
            raise ValueError("用户名或密码错误")
        token = secrets.token_urlsafe(36)
        expires_at = timestamp + SESSION_TTL_SECONDS * 1000
        conn.execute(
            "INSERT INTO sessions(token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token, row["id"], expires_at, timestamp),
        )
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (timestamp, row["id"]))
        return user_from_row(row, last_login_at=timestamp), token, expires_at


def logout_user(config: RagConfig, *, token: str) -> None:
    with connect_metadata_db(config) as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def authenticate_token(config: RagConfig, *, token: str | None) -> User | None:
    if not token:
        return None
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (timestamp,))
        row = conn.execute(
            """
            SELECT users.* FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ? AND sessions.expires_at > ?
            """,
            (token, timestamp),
        ).fetchone()
        if row is None:
            return None
        return user_from_row(row)


def list_public_users(config: RagConfig) -> list[User]:
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT id, username, display_name, role, tenant_id, created_at, last_login_at
            FROM users
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [user_from_row(row) for row in rows]


def create_announcement(
    config: RagConfig,
    *,
    title: str,
    content: str,
    author_id: str,
) -> dict[str, Any]:
    clean_title = " ".join(title.split()).strip()
    clean_content = content.strip()
    if not clean_title:
        raise ValueError("公告标题不能为空")
    if not clean_content:
        raise ValueError("公告内容不能为空")
    if len(clean_title) > 80:
        raise ValueError("公告标题不能超过 80 个字符")
    if len(clean_content) > 2000:
        raise ValueError("公告内容不能超过 2000 个字符")
    row = {
        "id": f"announcement-{uuid.uuid4().hex[:12]}",
        "title": clean_title,
        "content": clean_content,
        "author_id": author_id,
        "created_at": now_ms(),
    }
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO announcements(id, title, content, author_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (row["id"], row["title"], row["content"], row["author_id"], row["created_at"]),
        )
    return row


def list_announcements(config: RagConfig, *, limit: int = 5) -> list[dict[str, Any]]:
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT announcements.id, announcements.title, announcements.content,
                   announcements.author_id, announcements.created_at,
                   users.display_name AS author_name
            FROM announcements
            JOIN users ON users.id = announcements.author_id
            ORDER BY announcements.created_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 20)),),
        ).fetchall()
    return [dict(row) for row in rows]


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    return authorization[len(prefix) :].strip() or None


def user_from_row(row, *, last_login_at: int | None = None) -> User:
    return User(
        id=str(row["id"]),
        username=str(row["username"]),
        display_name=str(row["display_name"]),
        role=str(row["role"]),
        tenant_id=str(row["tenant_id"]),
        created_at=int(row["created_at"]),
        last_login_at=last_login_at if last_login_at is not None else row["last_login_at"],
    )
