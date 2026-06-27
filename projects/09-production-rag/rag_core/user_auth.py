from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
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
TEST_ACCOUNT_ID = "user-fixed-test"
TEST_ACCOUNT_USERNAME = "test_user"
TEST_ACCOUNT_PASSWORD = "12345678"
TEST_ACCOUNT_DISPLAY_NAME = "测试账号"
TEST_ACCOUNT_TENANT_ID = "tenant-fixed-test"
TEST_ACCOUNT_SALT = "0123456789abcdeffedcba9876543210"
LEGACY_TEST_ACCOUNT_CREATED_AT = 1704067200000
TEST_ACCOUNT_TOKEN_EXPIRES_AT = 4102444800000
_ANNOUNCEMENT_CACHE_LOCK = threading.Lock()
_ANNOUNCEMENT_CACHE: dict[tuple[str, int], tuple[float, list[dict[str, Any]]]] = {}
_AUTH_TOKEN_CACHE_LOCK = threading.Lock()
_AUTH_TOKEN_CACHE: dict[tuple[str, str], tuple[float, "User"]] = {}
_AUTH_TOKEN_CACHE_METRICS = {
    "hits_total": 0,
    "misses_total": 0,
    "invalidations_total": 0,
}


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
    profile_name_edit_allowed: bool = True
    avatar_edit_allowed: bool = True
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
            "profile_name_edit_allowed": self.profile_name_edit_allowed,
            "avatar_edit_allowed": self.avatar_edit_allowed,
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
        admin_count = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'").fetchone()[0]
        if admin_count > 0 and user_count > 0 and not registration_enabled_from_conn(conn):
            raise ValueError("新用户注册已被管理员关闭")
        role = "admin" if admin_count == 0 else "user"
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
        profile_name_edit_allowed=True,
        avatar_edit_allowed=True,
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
        token = config.fixed_test_login_token if str(row["username"]) == TEST_ACCOUNT_USERNAME else generate_session_token()
        expires_at = (
            TEST_ACCOUNT_TOKEN_EXPIRES_AT
            if str(row["username"]) == TEST_ACCOUNT_USERNAME
            else timestamp + SESSION_TTL_SECONDS * 1000
        )
        conn.execute(
            "INSERT OR REPLACE INTO sessions(token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token, row["id"], expires_at, timestamp),
        )
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (timestamp, row["id"]))
    invalidate_auth_token_cache(config)
    user = user_from_row(row, last_login_at=timestamp)
    cache_auth_token(config=config, token=token, user=user, expires_at_ms=expires_at)
    return user, token, expires_at


def generate_session_token(length: int = 24) -> str:
    return "".join(secrets.choice(SESSION_TOKEN_ALPHABET) for _ in range(length))


def refresh_session_token(config: RagConfig, *, current_token: str) -> tuple[User, str, int]:
    if current_token == config.fixed_test_login_token:
        raise ValueError("测试账号使用固定登录 token，不能刷新")
    timestamp = now_ms()
    expires_at = timestamp + SESSION_TTL_SECONDS * 1000
    with connect_metadata_db(config) as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (timestamp,))
        row = conn.execute(
            """
            SELECT users.* FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ? AND sessions.expires_at > ?
            """,
            (current_token, timestamp),
        ).fetchone()
        if row is None:
            raise ValueError("请先登录")
        if str(row["status"] or "active") != "active":
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
            raise ValueError("账号已被封禁")
        if str(row["username"]) == TEST_ACCOUNT_USERNAME:
            raise ValueError("测试账号使用固定登录 token，不能刷新")

        token = generate_session_token()
        while conn.execute("SELECT 1 FROM sessions WHERE token = ?", (token,)).fetchone():
            token = generate_session_token()
        conn.execute("DELETE FROM sessions WHERE token = ?", (current_token,))
        conn.execute(
            "INSERT INTO sessions(token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token, row["id"], expires_at, timestamp),
        )
    invalidate_auth_token_cache(config, token=current_token)
    user = user_from_row(row)
    cache_auth_token(config=config, token=token, user=user, expires_at_ms=expires_at)
    return user, token, expires_at


def logout_user(config: RagConfig, *, token: str) -> None:
    if token == config.fixed_test_login_token:
        invalidate_auth_token_cache(config, token=token)
        return
    with connect_metadata_db(config) as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    invalidate_auth_token_cache(config, token=token)


def ensure_default_test_account(config: RagConfig) -> User:
    password_hash = hash_password(TEST_ACCOUNT_PASSWORD, TEST_ACCOUNT_SALT)
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (TEST_ACCOUNT_USERNAME,)).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO users(
                    id, username, display_name, password_hash, salt,
                    role, tenant_id, avatar_url, status,
                    profile_name_edit_allowed, avatar_edit_allowed,
                    created_at, last_login_at
                )
                VALUES (?, ?, ?, ?, ?, 'user', ?, '', 'active', 1, 1, ?, NULL)
                """,
                (
                    TEST_ACCOUNT_ID,
                    TEST_ACCOUNT_USERNAME,
                    TEST_ACCOUNT_DISPLAY_NAME,
                    password_hash,
                    TEST_ACCOUNT_SALT,
                    TEST_ACCOUNT_TENANT_ID,
                    timestamp,
                ),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (TEST_ACCOUNT_ID,)).fetchone()
        elif int(row["created_at"]) == LEGACY_TEST_ACCOUNT_CREATED_AT:
            conn.execute(
                "UPDATE users SET created_at = ? WHERE id = ?",
                (timestamp, row["id"]),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone()
        if str(row["status"] or "active") == "active":
            conn.execute(
                "INSERT OR REPLACE INTO sessions(token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (config.fixed_test_login_token, row["id"], TEST_ACCOUNT_TOKEN_EXPIRES_AT, timestamp),
            )
        invalidate_auth_token_cache(config)
        return user_from_row(row)


def authenticate_token(config: RagConfig, *, token: str | None) -> User | None:
    if not token:
        return None
    cached = get_cached_auth_token(config=config, token=token)
    if cached is not None:
        return cached
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (timestamp,))
        row = conn.execute(
            """
            SELECT users.*, sessions.expires_at AS session_expires_at FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ? AND sessions.expires_at > ?
            """,
            (token, timestamp),
        ).fetchone()
        if row is None:
            return None
        if str(row["status"] or "active") != "active":
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
            invalidate_auth_token_cache(config)
            return None
        if token == config.fixed_test_login_token and str(row["username"]) == TEST_ACCOUNT_USERNAME:
            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (timestamp, row["id"]))
            user = user_from_row(row, last_login_at=timestamp)
            cache_auth_token(config=config, token=token, user=user, expires_at_ms=int(row["session_expires_at"]))
            return user
        user = user_from_row(row)
        cache_auth_token(config=config, token=token, user=user, expires_at_ms=int(row["session_expires_at"]))
        return user


def get_cached_auth_token(*, config: RagConfig, token: str) -> User | None:
    ttl_seconds = auth_token_cache_ttl_seconds()
    if ttl_seconds <= 0:
        with _AUTH_TOKEN_CACHE_LOCK:
            _AUTH_TOKEN_CACHE_METRICS["misses_total"] += 1
        return None
    key = auth_token_cache_key(config=config, token=token)
    now = time.monotonic()
    with _AUTH_TOKEN_CACHE_LOCK:
        cached = _AUTH_TOKEN_CACHE.get(key)
        if cached is not None and cached[0] > now:
            _AUTH_TOKEN_CACHE_METRICS["hits_total"] += 1
            return cached[1]
        if cached is not None:
            _AUTH_TOKEN_CACHE.pop(key, None)
        _AUTH_TOKEN_CACHE_METRICS["misses_total"] += 1
    return None


def cache_auth_token(*, config: RagConfig, token: str, user: User, expires_at_ms: int) -> None:
    ttl_seconds = auth_token_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return
    session_remaining_seconds = max(0.0, (expires_at_ms - now_ms()) / 1000)
    ttl = min(ttl_seconds, session_remaining_seconds)
    if ttl <= 0:
        return
    key = auth_token_cache_key(config=config, token=token)
    with _AUTH_TOKEN_CACHE_LOCK:
        _AUTH_TOKEN_CACHE[key] = (time.monotonic() + ttl, user)


def auth_token_cache_ttl_seconds() -> float:
    value = os.environ.get("RAG_AUTH_TOKEN_CACHE_TTL_SECONDS", "2")
    try:
        return max(0.0, float(value))
    except ValueError:
        return 2.0


def auth_token_cache_key(*, config: RagConfig, token: str) -> tuple[str, str]:
    db_key = config.metadata_database_url or str(config.runtime_dir / "production_rag.db")
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return (db_key, token_hash)


def invalidate_auth_token_cache(config: RagConfig | None = None, *, token: str | None = None) -> None:
    with _AUTH_TOKEN_CACHE_LOCK:
        _AUTH_TOKEN_CACHE_METRICS["invalidations_total"] += 1
        if config is None:
            _AUTH_TOKEN_CACHE.clear()
            return
        db_key = config.metadata_database_url or str(config.runtime_dir / "production_rag.db")
        if token is not None:
            _AUTH_TOKEN_CACHE.pop(auth_token_cache_key(config=config, token=token), None)
            return
        for key in list(_AUTH_TOKEN_CACHE):
            if key[0] == db_key:
                _AUTH_TOKEN_CACHE.pop(key, None)


def auth_token_cache_metrics_snapshot() -> dict[str, int | float]:
    with _AUTH_TOKEN_CACHE_LOCK:
        return {
            **_AUTH_TOKEN_CACHE_METRICS,
            "entries": len(_AUTH_TOKEN_CACHE),
            "ttl_seconds": auth_token_cache_ttl_seconds(),
        }


def list_public_users(config: RagConfig, *, query: str = "", limit: int = 50, offset: int = 0) -> list[User]:
    clean_query = query.strip().lower()
    clean_limit = max(1, min(limit, 100))
    clean_offset = max(0, offset)
    with connect_metadata_db(config) as conn:
        params: list[Any] = []
        where_sql = ""
        if clean_query:
            where_sql = "WHERE lower(username) LIKE ? OR lower(display_name) LIKE ? OR tenant_id LIKE ?"
            like_query = f"%{clean_query}%"
            params.extend([like_query, like_query, like_query])
        rows = conn.execute(
            f"""
            SELECT id, username, display_name, role, tenant_id, avatar_url, status,
                   profile_name_edit_allowed, avatar_edit_allowed, created_at, last_login_at
            FROM users
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, clean_limit, clean_offset),
        ).fetchall()
    return [user_from_row(row) for row in rows]


def count_public_users(config: RagConfig, *, query: str = "") -> int:
    clean_query = query.strip().lower()
    with connect_metadata_db(config) as conn:
        if clean_query:
            like_query = f"%{clean_query}%"
            row = conn.execute(
                """
                SELECT COUNT(*) FROM users
                WHERE lower(username) LIKE ? OR lower(display_name) LIKE ? OR tenant_id LIKE ?
                """,
                (like_query, like_query, like_query),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
    return int(row[0])


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
        current = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if current is None:
            raise ValueError("用户不存在")
        name_changed = username != str(current["username"]) or display != str(current["display_name"])
        avatar_changed = avatar != str(current["avatar_url"] or "")
        if name_changed and int(current["profile_name_edit_allowed"] or 0) != 1:
            raise ValueError("管理员已关闭名称修改权限")
        if avatar_changed and int(current["avatar_edit_allowed"] or 0) != 1:
            raise ValueError("管理员已关闭头像修改权限")
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
    invalidate_auth_token_cache(config)
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
    invalidate_auth_token_cache(config)


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
    invalidate_auth_token_cache(config)
    return user_from_row(updated)


def bulk_update_users(config: RagConfig, *, actor_id: str, updates: list[dict[str, Any]]) -> list[User]:
    if not updates:
        raise ValueError("请选择要更新的用户")
    if len(updates) > 50:
        raise ValueError("一次最多更新 50 个用户")
    updated_users: list[User] = []
    with connect_metadata_db(config) as conn:
        for update in updates:
            user_id = str(update.get("user_id") or "").strip()
            if not user_id:
                raise ValueError("用户 ID 不能为空")
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise ValueError("用户不存在")
            if str(row["role"]) == "admin" and user_id != actor_id:
                raise ValueError("不能批量修改其他管理员账号")

            status = str(row["status"] or "active")
            profile_name_edit_allowed = bool(int(row["profile_name_edit_allowed"] or 0))
            avatar_edit_allowed = bool(int(row["avatar_edit_allowed"] or 0))

            if "status" in update and update["status"] is not None:
                status = str(update["status"])
                if status not in {"active", "banned"}:
                    raise ValueError("用户状态无效")
                if user_id == actor_id and status == "banned":
                    raise ValueError("不能封禁当前管理员账号")
                if str(row["role"]) == "admin" and status == "banned":
                    raise ValueError("不能封禁管理员账号")
            if "profile_name_edit_allowed" in update and update["profile_name_edit_allowed"] is not None:
                profile_name_edit_allowed = bool(update["profile_name_edit_allowed"])
            if "avatar_edit_allowed" in update and update["avatar_edit_allowed"] is not None:
                avatar_edit_allowed = bool(update["avatar_edit_allowed"])

            conn.execute(
                """
                UPDATE users
                SET status = ?, profile_name_edit_allowed = ?, avatar_edit_allowed = ?
                WHERE id = ?
                """,
                (
                    status,
                    1 if profile_name_edit_allowed else 0,
                    1 if avatar_edit_allowed else 0,
                    user_id,
                ),
            )
            if status == "banned":
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            updated_users.append(user_from_row(updated))
    invalidate_auth_token_cache(config)
    return updated_users


def create_announcement(
    config: RagConfig,
    *,
    title: str,
    content: str,
    author_id: str,
    link_url: str | None = None,
    link_label: str | None = None,
) -> dict[str, Any]:
    clean_title = " ".join(title.split()).strip()
    clean_content = content.strip()
    clean_link_url = validate_announcement_link_url(link_url)
    clean_link_label = " ".join((link_label or "").split()).strip()[:80]
    if clean_link_url and not clean_link_label:
        clean_link_label = "查看详情"
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
        "link_url": clean_link_url,
        "link_label": clean_link_label,
        "author_id": author_id,
        "created_at": now_ms(),
    }
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO announcements(id, title, content, link_url, link_label, author_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["title"],
                row["content"],
                row["link_url"],
                row["link_label"],
                row["author_id"],
                row["created_at"],
            ),
        )
    invalidate_announcement_cache(config)
    return row


def validate_announcement_link_url(link_url: str | None) -> str:
    value = (link_url or "").strip()
    if len(value) > 500:
        raise ValueError("公告链接不能超过 500 个字符")
    if value and not (value.startswith("http://") or value.startswith("https://") or value.startswith("/")):
        raise ValueError("公告链接需要是 http(s) URL 或站内路径")
    return value


def list_announcements(config: RagConfig, *, limit: int = 5) -> list[dict[str, Any]]:
    clean_limit = max(1, min(limit, 20))
    ttl_seconds = announcement_cache_ttl_seconds()
    if ttl_seconds > 0:
        cache_key = announcement_cache_key(config=config, limit=clean_limit)
        now = time.monotonic()
        with _ANNOUNCEMENT_CACHE_LOCK:
            cached = _ANNOUNCEMENT_CACHE.get(cache_key)
            if cached is not None and cached[0] > now:
                return [dict(row) for row in cached[1]]
    else:
        cache_key = None

    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT announcements.id, announcements.title, announcements.content,
                   announcements.link_url, announcements.link_label,
                   announcements.author_id, announcements.created_at,
                   users.display_name AS author_name
            FROM announcements
            JOIN users ON users.id = announcements.author_id
            ORDER BY announcements.created_at DESC
            LIMIT ?
            """,
            (clean_limit,),
        ).fetchall()
    result = [dict(row) for row in rows]
    if ttl_seconds > 0 and cache_key is not None:
        with _ANNOUNCEMENT_CACHE_LOCK:
            _ANNOUNCEMENT_CACHE[cache_key] = (time.monotonic() + ttl_seconds, [dict(row) for row in result])
    return result


def delete_announcement(config: RagConfig, *, announcement_id: str) -> bool:
    with connect_metadata_db(config) as conn:
        cursor = conn.execute("DELETE FROM announcements WHERE id = ?", (announcement_id,))
        removed = cursor.rowcount > 0
    if removed:
        invalidate_announcement_cache(config)
    return removed


def announcement_cache_ttl_seconds() -> float:
    value = os.environ.get("RAG_ANNOUNCEMENT_CACHE_TTL_SECONDS", "5")
    try:
        return max(0.0, float(value))
    except ValueError:
        return 5.0


def announcement_cache_key(*, config: RagConfig, limit: int) -> tuple[str, int]:
    db_key = config.metadata_database_url or str(config.runtime_dir / "production_rag.db")
    return (db_key, limit)


def invalidate_announcement_cache(config: RagConfig | None = None) -> None:
    with _ANNOUNCEMENT_CACHE_LOCK:
        if config is None:
            _ANNOUNCEMENT_CACHE.clear()
            return
        db_key = config.metadata_database_url or str(config.runtime_dir / "production_rag.db")
        for key in list(_ANNOUNCEMENT_CACHE):
            if key[0] == db_key:
                _ANNOUNCEMENT_CACHE.pop(key, None)


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
        profile_name_edit_allowed=bool(int(row["profile_name_edit_allowed"] or 0)),
        avatar_edit_allowed=bool(int(row["avatar_edit_allowed"] or 0)),
        last_login_at=last_login_at if last_login_at is not None else row["last_login_at"],
    )
