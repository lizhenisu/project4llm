from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import serve


@dataclass(frozen=True)
class SyntheticTrace:
    request_id: str
    retrieval_mode: str


def resolve_synthetic_answer(request, _auth_context, stage_callback=None):
    coordination_dir = Path(os.environ["RAG_QUERY_RECOVERY_COORDINATION_DIR"])
    coordination_dir.mkdir(parents=True, exist_ok=True)
    call_marker = coordination_dir / "model-call"
    try:
        descriptor = os.open(call_marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Synthetic model work ran more than once") from exc
    else:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))

    if stage_callback is not None:
        stage_callback(
            {
                "stage": "search",
                "status": "active",
                "label": "向量检索",
                "detail": "跨实例检索已经开始。",
            }
        )
    (coordination_dir / "started").touch()

    deadline = time.monotonic() + 15
    release_marker = coordination_dir / "release"
    while not release_marker.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Synthetic query was not released")
        time.sleep(0.02)

    if stage_callback is not None:
        stage_callback(
            {
                "stage": "search",
                "status": "done",
                "label": "向量检索",
                "detail": "跨实例检索已经完成。",
            }
        )
    return SimpleNamespace(
        request_id=request.request_id,
        answer="跨实例恢复只执行了一次模型工作。",
        hits=[],
        candidates=[],
        reranked=[],
        trace=SyntheticTrace(
            request_id=request.request_id,
            retrieval_mode="synthetic-multi-instance-recovery",
        ),
        generation={"model": "synthetic"},
    )


serve.resolve_answer_result = resolve_synthetic_answer
app = serve.create_app()
