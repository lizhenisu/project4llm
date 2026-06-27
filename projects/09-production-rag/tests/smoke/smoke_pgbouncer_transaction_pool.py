from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import psycopg


CLIENT_CONNECTIONS = 40
EXPECTED_BACKEND_LIMIT = 20


def main() -> None:
    database_url = os.environ.get("SMOKE_METADATA_DATABASE_URL", "")
    if not database_url:
        raise RuntimeError("SMOKE_METADATA_DATABASE_URL is required for this smoke test.")

    connections: list[psycopg.Connection] = []
    try:
        for _ in range(CLIENT_CONNECTIONS):
            connections.append(psycopg.connect(database_url, autocommit=True))

        with ThreadPoolExecutor(max_workers=CLIENT_CONNECTIONS) as executor:
            backend_pids = list(executor.map(query_backend_pid, connections))

        distinct_backend_pids = set(backend_pids)
        assert len(distinct_backend_pids) <= EXPECTED_BACKEND_LIMIT, (
            f"{CLIENT_CONNECTIONS} client sessions used {len(distinct_backend_pids)} PostgreSQL backends; "
            f"expected at most {EXPECTED_BACKEND_LIMIT} through transaction pooling"
        )
    finally:
        for connection in connections:
            connection.close()

    print(
        "smoke_pgbouncer_transaction_pool=ok "
        f"clients={CLIENT_CONNECTIONS} backends={len(distinct_backend_pids)}"
    )


def query_backend_pid(connection: psycopg.Connection) -> int:
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_backend_pid() FROM pg_sleep(0.15)")
        row = cursor.fetchone()
    assert row is not None
    return int(row[0])


if __name__ == "__main__":
    main()
