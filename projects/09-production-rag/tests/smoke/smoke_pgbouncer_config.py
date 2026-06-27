from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
START_SCRIPT = PROJECT_DIR / "ops" / "pgbouncer" / "start_pgbouncer.py"


def render_config(config_dir: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "POSTGRES_HOST": "postgres",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": 'synthetic\\"user',
        "POSTGRES_PASSWORD": 'synthetic\\"password',
        "PGBOUNCER_CONFIG_DIR": str(config_dir),
        **overrides,
    }
    runner = (
        "import os,runpy;"
        "os.execvp=lambda *args: None;"
        f"runpy.run_path({str(START_SCRIPT)!r},run_name='__main__')"
    )
    return subprocess.run(
        [sys.executable, "-c", runner],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rag-pgbouncer-") as temp_dir:
        config_dir = Path(temp_dir)
        result = render_config(config_dir)
        assert result.returncode == 0, result.stderr

        config_path = config_dir / "pgbouncer.ini"
        userlist_path = config_dir / "userlist.txt"
        config = config_path.read_text(encoding="utf-8")
        userlist = userlist_path.read_text(encoding="utf-8")

        assert "* = host=postgres port=5432" in config
        assert "listen_port = 6432" in config
        assert "auth_type = scram-sha-256" in config
        assert "pool_mode = transaction" in config
        assert "max_client_conn = 200" in config
        assert "default_pool_size = 20" in config
        assert "reserve_pool_size = 5" in config
        assert "max_db_connections = 60" in config
        assert "synthetic" not in config
        assert userlist == '"synthetic\\\\\\"user" "synthetic\\\\\\"password"\n'
        assert config_path.stat().st_mode & 0o777 == 0o600
        assert userlist_path.stat().st_mode & 0o777 == 0o600

        invalid_result = render_config(config_dir, PGBOUNCER_MAX_CLIENT_CONN="0")
        assert invalid_result.returncode != 0
        assert "PGBOUNCER_MAX_CLIENT_CONN must be greater than zero" in invalid_result.stderr

        injection_result = render_config(config_dir, POSTGRES_PASSWORD="bad\npassword")
        assert injection_result.returncode != 0
        assert "unsupported control character" in injection_result.stderr

    print("smoke_pgbouncer_config=ok")


if __name__ == "__main__":
    main()
