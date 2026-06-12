from __future__ import annotations

import argparse
from pathlib import Path

from rag_core.io import read_jsonl
from rag_core.pii import detect_pii


TEXT_FIELDS = ("text", "ocr_text", "caption")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan JSONL knowledge files for PII.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--fail", action="store_true", help="Exit non-zero if PII is found.")
    args = parser.parse_args()

    total = 0
    for path in args.paths:
        rows = read_jsonl(path)
        for index, row in enumerate(rows, start=1):
            for field in TEXT_FIELDS:
                value = row.get(field)
                if not isinstance(value, str):
                    continue
                findings = detect_pii(value)
                for finding in findings:
                    total += 1
                    print(f"{path}:{index}:{field}: {finding.kind}")

    print(f"pii_findings: {total}")
    if args.fail and total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

