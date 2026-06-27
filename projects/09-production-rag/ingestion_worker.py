from __future__ import annotations

import signal
import threading

from rag_core.ingestion_jobs import run_ingestion_worker


def main() -> None:
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    run_ingestion_worker(stop_event=stop_event)


if __name__ == "__main__":
    main()
