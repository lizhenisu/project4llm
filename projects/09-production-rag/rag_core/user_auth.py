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
REGISTRATION_ENABLED_KEY = "registration_enabled"
SESSION_TOKEN_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class User:
    id: str
    username: str
    display_name: str
    role: str
    tenant_id: str
    created_at: int
    avatar_url: str = ""
    status: str = "active"
    last_login_at: int | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "avatar_url": self.avatar_url,
            "status": self.status,
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


def validate_avatar_url(avatar_url: str | None) -> str:
    value = (avatar_url or "").strip()
    if len(value) > 500:
        raise ValueError("头像地址不能超过 500 个字符")
    if value and not (value.startswith("http://") or value.startswith("https://") or value.startswith("data:image/")):
        raise ValueError("头像地址需要是 http(s) URL 或 data:image")
    return value


def normalize_display_name(display_name: str | None, fallback: str) -> str:
    return (display_name or fallback).strip()[:40] or fallback


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
    display = normalize_display_name(display_name, username)
    with connect_metadata_db(config) as conn:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if user_count > 0 and not registration_enabled_from_conn(conn):
            raise ValueError("新用户注册已被管理员关闭")
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
        avatar_url="",
        status="active",
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
        if str(row["status"] or "active") != "active":
            raise ValueError("账号已被封禁")
        token = generate_session_token()
        expires_at = timestamp + SESSION_TTL_SECONDS * 1000
        conn.execute(
            "INSERT INTO sessions(token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token, row["id"], expires_at, timestamp),
        )
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (timestamp, row["id"]))
        return user_from_row(row, last_login_at=timestamp), token, expires_at


def generate_session_token(length: int = 24) -> str:
    return "".join(secrets.choice(SESSION_TOKEN_ALPHABET) for _ in range(length))


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
        if str(row["status"] or "active") != "active":
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
            return None
        return user_from_row(row)


def list_public_users(config: RagConfig) -> list[User]:
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT id, username, display_name, role, tenant_id, avatar_url, status, created_at, last_login_at
            FROM users
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [user_from_row(row) for row in rows]


def update_user_profile(
    config: RagConfig,
    *,
    user_id: str,
    username: str,
    display_name: str,
    avatar_url: str | None,
) -> User:
    username = normalize_username(username)
    display = normalize_display_name(display_name, username)
    avatar = validate_avatar_url(avatar_url)
    with connect_metadata_db(config) as conn:
        try:
            conn.execute(
                """
                UPDATE users
                SET username = ?, display_name = ?, avatar_url = ?
                WHERE id = ?
                """,
                (username, display, avatar, user_id),
            )
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                raise ValueError("用户名已存在") from exc
            raise
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise ValueError("用户不存在")
    return user_from_row(row)


def change_user_password(
    config: RagConfig,
    *,
    user_id: str,
    current_password: str,
    new_password: str,
) -> None:
    validate_password(new_password)
    with connect_metadata_db(config) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None or not verify_password(
            current_password,
            salt=str(row["salt"]),
            password_hash=str(row["password_hash"]),
        ):
            raise ValueError("当前密码错误")
        salt = secrets.token_hex(16)
        password_hash = hash_password(new_password, salt)
        conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
            (password_hash, salt, user_id),
        )


def set_user_status(config: RagConfig, *, actor_id: str, user_id: str, status: str) -> User:
    if status not in {"active", "banned"}:
        raise ValueError("用户状态无效")
    if actor_id == user_id:
        raise ValueError("不能封禁当前管理员账号")
    with connect_metadata_db(config) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise ValueError("用户不存在")
        if str(row["role"]) == "admin":
            raise ValueError("不能封禁管理员账号")
        conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
        if status == "banned":
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return user_from_row(updated)


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


def is_registration_enabled(config: RagConfig) -> bool:
    with connect_metadata_db(config) as conn:
        return registration_enabled_from_conn(conn)


def set_registration_enabled(config: RagConfig, *, enabled: bool) -> bool:
    value = "1" if enabled else "0"
    with connect_metadata_db(config) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            (REGISTRATION_ENABLED_KEY, value),
        )
    return enabled


def registration_enabled_from_conn(conn) -> bool:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = ?",
        (REGISTRATION_ENABLED_KEY,),
    ).fetchone()
    if row is None:
        return True
    return str(row["value"]) != "0"


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
        avatar_url=str(row["avatar_url"] or ""),
        status=str(row["status"] or "active"),
        last_login_at=last_login_at if last_login_at is not None else row["last_login_at"],
    )
