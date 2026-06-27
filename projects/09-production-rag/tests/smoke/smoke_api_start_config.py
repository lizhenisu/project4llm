from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
START_SCRIPT = PROJECT_DIR / "scripts" / "start_api.sh"


def main() -> None:
    test_start_script_forwards_runtime_limits()
    test_start_script_rejects_invalid_worker_count()
    print("smoke_api_start_config=ok")


def test_start_script_forwards_runtime_limits() -> None:
    result = run_with_fake_commands(
        {
            "RAG_API_PORT": "9010",
            "RAG_API_WORKERS": "3",
            "RAG_API_LIMIT_CONCURRENCY": "480",
            "RAG_API_KEEP_ALIVE_SECONDS": "20",
            "RAG_API_GRACEFUL_SHUTDOWN_SECONDS": "45",
        }
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.strip().splitlines()[-1]
    assert args == (
        "serve:app --host 0.0.0.0 --port 9010 --workers 3 "
        "--limit-concurrency 480 --timeout-keep-alive 20 "
        "--timeout-graceful-shutdown 45"
    )


def test_start_script_rejects_invalid_worker_count() -> None:
    result = run_with_fake_commands({"RAG_API_WORKERS": "0"})
    assert result.returncode == 2
    assert "RAG_API_WORKERS must be a positive integer" in result.stderr
    assert result.stdout == ""


def run_with_fake_commands(overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        bin_dir = Path(tmp)
        write_executable(bin_dir / "python", "#!/bin/sh\nexit 0\n")
        write_executable(bin_dir / "uvicorn", '#!/bin/sh\nprintf "%s\\n" "$*"\n')
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            **overrides,
        }
        return subprocess.run(
            ["bash", str(START_SCRIPT)],
            cwd=PROJECT_DIR,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


if __name__ == "__main__":
    main()
