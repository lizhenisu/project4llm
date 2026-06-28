from __future__ import annotations

import os
from pathlib import Path


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} must not be empty")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise SystemExit(f"{name} contains an unsupported control character")
    return value


def positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be greater than zero")
    return value


def auth_file_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    postgres_host = required_env("POSTGRES_HOST")
    postgres_port = positive_int_env("POSTGRES_PORT", 5432)
    postgres_user = required_env("POSTGRES_USER")
    postgres_password = required_env("POSTGRES_PASSWORD")
    config_dir = Path(os.environ.get("PGBOUNCER_CONFIG_DIR", "/var/run/pgbouncer-config"))
    config_dir.mkdir(parents=True, exist_ok=True)

    userlist_path = config_dir / "userlist.txt"
    userlist_path.write_text(
        f'"{auth_file_value(postgres_user)}" "{auth_file_value(postgres_password)}"\n',
        encoding="utf-8",
    )
    userlist_path.chmod(0o600)

    config_path = config_dir / "pgbouncer.ini"
    config_path.write_text(
        "\n".join(
            (
                "[databases]",
                f"* = host={postgres_host} port={postgres_port}",
                "",
                "[pgbouncer]",
                "listen_addr = 0.0.0.0",
                "listen_port = 6432",
                "unix_socket_dir =",
                "auth_type = scram-sha-256",
                f"auth_file = {userlist_path}",
                "pool_mode = transaction",
                f"max_client_conn = {positive_int_env('PGBOUNCER_MAX_CLIENT_CONN', 200)}",
                f"default_pool_size = {positive_int_env('PGBOUNCER_DEFAULT_POOL_SIZE', 20)}",
                f"reserve_pool_size = {positive_int_env('PGBOUNCER_RESERVE_POOL_SIZE', 5)}",
                f"max_db_connections = {positive_int_env('PGBOUNCER_MAX_DB_CONNECTIONS', 60)}",
                f"server_idle_timeout = {positive_int_env('PGBOUNCER_SERVER_IDLE_TIMEOUT_SECONDS', 600)}",
                f"query_wait_timeout = {positive_int_env('PGBOUNCER_QUERY_WAIT_TIMEOUT_SECONDS', 30)}",
                "log_connections = 1",
                "log_disconnections = 1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    os.execvp("pgbouncer", ("pgbouncer", str(config_path)))


if __name__ == "__main__":
    main()
