from __future__ import annotations

from rag_core.config import load_config
from rag_core.readiness import readiness_report, redacted_config


def main() -> None:
    config = load_config()
    print("RAG config:")
    for key, value in redacted_config(config).items():
        print(f"- {key}: {value}")

    report = readiness_report(config)
    print("Readiness:")
    for name, check in report["checks"].items():
        print(f"- {name}: {check}")
    if report["status"] != "ok":
        raise SystemExit("RAG readiness check failed")


if __name__ == "__main__":
    main()
