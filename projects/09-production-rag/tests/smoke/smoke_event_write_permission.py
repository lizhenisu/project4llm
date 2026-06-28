from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.events import append_event


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime_dir = Path(tmp)
        blocked = runtime_dir / "answer_events.jsonl"
        blocked.write_text("", encoding="utf-8")
        with patch("pathlib.Path.open", side_effect=PermissionError("permission denied")):
            append_event(runtime_dir, "answer_events", {"question": "hello"})
        assert blocked.read_text(encoding="utf-8") == ""
    print("smoke_event_write_permission=ok")


if __name__ == "__main__":
    main()
