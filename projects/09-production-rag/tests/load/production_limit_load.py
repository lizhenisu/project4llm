from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = PROJECT_DIR.parents[1]
RUNNER = PROJECT_DIR / "tests" / "load" / "rag_query_load.py"
DEFAULT_SOURCE_TYPES = ["pdf", "txt", "md", "html", "table", "csv", "tsv"]
DEFAULT_LEVELS = [1, 2, 4, 8, 16, 32, 64]


@dataclass(frozen=True)
class StageResult:
    concurrency: int
    summary: dict[str, Any]
    stdout_path: Path
    stderr_path: Path
    json_path: Path
    usable: bool
    reasons: list[str]


def main() -> None:
    args = parse_args()
    started_at = datetime.now().astimezone()
    run_id = started_at.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or PROJECT_DIR / "docs" / "load-tests" / f"production-limit-{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = parse_levels(args.concurrency_levels)

    if args.dry_run:
        print_dry_run(args, output_dir=output_dir, levels=levels)
        return

    health = check_health(args.base_url, timeout=args.health_timeout)
    version = detect_version(args.base_url)
    inventory = collect_source_inventory(args)
    enforce_rag_inventory(inventory)
    metadata = collect_metadata(args, version=version, started_at=started_at, inventory=inventory)

    results: list[StageResult] = []
    print(f"Load-test target: {args.base_url}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Health: {health}", flush=True)
    if args.search_limit:
        print(
            "Concurrency search: "
            f"{args.search_min_concurrency}..{args.search_max_concurrency} "
            f"(precision {args.search_precision})",
            flush=True,
        )
    else:
        print(f"Concurrency levels: {', '.join(str(level) for level in levels)}", flush=True)
    for warning in inventory.get("warnings") or []:
        print(f"Data inventory warning: {warning}", flush=True)
    print(flush=True)

    if args.search_limit:
        results = run_search_limit(args, output_dir=output_dir)
    else:
        for concurrency in levels:
            result = run_and_print_stage(args, output_dir=output_dir, concurrency=concurrency)
            results.append(result)
            if args.stop_after_first_unusable and not result.usable:
                break

    report_path = output_dir / "report.md"
    report_path.write_text(
        render_report(
            args=args,
            metadata=metadata,
            health=health,
            results=results,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )
    machine_path = output_dir / "summary.json"
    machine_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "health": health,
                "results": [
                    {
                        "concurrency": result.concurrency,
                        "usable": result.usable,
                        "reasons": result.reasons,
                        "summary": result.summary,
                    }
                    for result in results
                ],
                "best_usable": best_usable_payload(results),
                "limit_summary": limit_summary_payload(results),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(flush=True)
    print_final_summary(results, search_limit=args.search_limit)
    print(f"Report: {report_path}", flush=True)
    print(f"Summary JSON: {machine_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ramp-test a deployed Production RAG service and generate a report. "
            "This is intended for real 2C2G container deployments."
        )
    )
    parser.add_argument("--base-url", default=os.environ.get("RAG_LOAD_BASE_URL", "http://127.0.0.1:8008"))
    parser.add_argument("--endpoint", default="/query/stream")
    parser.add_argument(
        "--token",
        default=os.environ.get("RAG_LOAD_TEST_TOKEN", "production-rag-fixed-test-login-token"),
        help="Bearer token for the deployed API. Defaults to RAG_LOAD_TEST_TOKEN or production-rag-fixed-test-login-token.",
    )
    parser.add_argument("--tenant-id", default=os.environ.get("RAG_LOAD_TENANT_ID", "team_a"))
    parser.add_argument("--acl-group", action="append", default=[])
    parser.add_argument(
        "--source-type",
        action="append",
        default=[],
        help="Restrict retrieval to source types. Defaults to common text/document types.",
    )
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--doc-version", type=int)
    parser.add_argument(
        "--current-only",
        action="store_false",
        dest="include_all_sources",
        default=True,
        help=(
            "Use only current documents. By default load tests use all visible ready sources."
        ),
    )
    parser.add_argument(
        "--include-source-identifiers",
        action="store_true",
        help=(
            "Include a limited doc_id/doc_version inventory in the report. "
            "Disabled by default because doc_id may contain filenames or private labels."
        ),
    )
    parser.add_argument(
        "--source-identifier-limit",
        type=int,
        default=20,
        help="Maximum source identifiers to include when --include-source-identifiers is set.",
    )
    parser.add_argument("--question", default=os.environ.get("RAG_LOAD_QUESTION", "总结这些资料的核心内容"))
    parser.add_argument("--questions-file", type=Path)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--context-limit", type=int, default=5)
    parser.add_argument(
        "--concurrency-levels",
        default=",".join(str(level) for level in DEFAULT_LEVELS),
        help="Comma-separated concurrency levels, default: 1,2,4,8,16,32,64.",
    )
    parser.add_argument(
        "--search-limit",
        action="store_true",
        help="Find an approximate concurrency limit with exponential probing and binary search.",
    )
    parser.add_argument("--search-min-concurrency", type=int, default=1)
    parser.add_argument("--search-max-concurrency", type=int, default=64)
    parser.add_argument(
        "--search-precision",
        type=int,
        default=1,
        help="Stop binary search when the unusable/usable boundary is within this concurrency gap.",
    )
    parser.add_argument("--requests-per-concurrency", type=int, default=5)
    parser.add_argument("--min-requests", type=int, default=20)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Maximum requests per concurrency level. Use 0 to disable the cap. Default: 0.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--health-timeout", type=float, default=5.0)
    parser.add_argument(
        "--inventory-timeout",
        type=float,
        default=15.0,
        help="Timeout for the preflight /sources request used to verify RAG data.",
    )
    parser.add_argument(
        "--inventory-retries",
        type=int,
        default=3,
        help="Retry count for the preflight /sources request before aborting.",
    )
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    parser.add_argument("--max-p95-ms", type=float, default=30000.0)
    parser.add_argument("--max-first-event-p95-ms", type=float, default=5000.0)
    parser.add_argument(
        "--min-throughput-rps",
        type=float,
        default=0.0,
        help="Optional lower bound for usable throughput.",
    )
    parser.add_argument(
        "--allow-zero-citations",
        action="store_true",
        help="Allow stages with no citations. By default this script requires citations to avoid direct-chat false positives.",
    )
    parser.add_argument(
        "--stop-after-first-unusable",
        action="store_true",
        help="Stop after the first unusable level. Default is to keep pushing through all configured levels.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--server-profile", default="2C2G")
    parser.add_argument(
        "--notes",
        default="",
        help="Free-form note included in the generated report.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_levels(raw: str) -> list[int]:
    levels = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise SystemExit("concurrency levels must be positive integers")
        levels.append(value)
    if not levels:
        raise SystemExit("no concurrency levels configured")
    return levels


def check_health(base_url: str, *, timeout: float) -> dict[str, Any]:
    url = urljoin(base_url.rstrip("/") + "/", "health")
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body else {}
            return {"ok": response.status == 200, "status_code": response.status, "body": parsed}
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Health check failed for {url}: {exc}") from exc


def detect_version(base_url: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    openapi_url = urljoin(base_url.rstrip("/") + "/", "openapi.json")
    try:
        with urlopen(Request(openapi_url, method="GET"), timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
            info = body.get("info") or {}
            if info.get("version"):
                payload["api_version"] = str(info["version"])
    except Exception:
        pass

    version_file = PROJECT_DIR / "VERSION"
    if version_file.exists():
        payload["local_version"] = version_file.read_text(encoding="utf-8").strip()

    git_commit = run_command(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR)
    if git_commit:
        payload["git_commit"] = git_commit
    git_branch = run_command(["git", "branch", "--show-current"], cwd=REPO_DIR)
    if git_branch:
        payload["git_branch"] = git_branch
    git_status = run_command(["git", "status", "--short"], cwd=REPO_DIR)
    payload["worktree"] = "dirty" if git_status else "clean"
    return payload


def collect_metadata(
    args: argparse.Namespace,
    *,
    version: dict[str, str],
    started_at: datetime,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    return {
        "started_at": started_at.isoformat(timespec="seconds"),
        "server_profile": args.server_profile,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "version": version,
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "mem_total_mb": mem_total_mb(),
        },
        "thresholds": {
            "max_failure_rate": args.max_failure_rate,
            "max_p95_ms": args.max_p95_ms,
            "max_first_event_p95_ms": args.max_first_event_p95_ms,
            "min_throughput_rps": args.min_throughput_rps,
            "require_citations": not args.allow_zero_citations,
        },
        "search": {
            "enabled": bool(args.search_limit),
            "min_concurrency": args.search_min_concurrency,
            "max_concurrency": args.search_max_concurrency,
            "precision": args.search_precision,
        },
        "request": {
            "question": args.question,
            "questions_file": str(args.questions_file) if args.questions_file else "",
            "source_types": selected_source_types(args),
            "doc_ids": args.doc_id,
            "doc_version": args.doc_version,
            "include_all_sources": args.include_all_sources,
            "candidate_limit": args.candidate_limit,
            "context_limit": args.context_limit,
        },
        "source_inventory": inventory,
        "notes": args.notes,
    }


def collect_source_inventory(args: argparse.Namespace) -> dict[str, Any]:
    query = urlencode({"tenant_id": args.tenant_id})
    url = urljoin(args.base_url.rstrip("/") + "/", f"sources?{query}")
    headers = {"Accept": "application/json", "X-RAG-Tenant-ID": args.tenant_id}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    if args.acl_group:
        headers["X-RAG-ACL-Groups"] = ",".join(args.acl_group)

    body: dict[str, Any] = {}
    last_error: Exception | None = None
    attempts = max(1, int(args.inventory_retries))
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers=headers, method="GET"), timeout=args.inventory_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2)
    else:
        error = str(last_error or "unknown inventory error")
        return {
            "ok": False,
            "error": error,
            "warnings": [f"Could not fetch /sources inventory: {error}"],
            "total_sources": 0,
            "in_scope_sources": 0,
            "ready_in_scope_sources": 0,
            "total_chunks": 0,
            "in_scope_chunks": 0,
            "by_type": {},
            "in_scope_by_type": {},
            "by_status": {},
            "in_scope_by_status": {},
        }

    sources = list(body.get("sources") or [])
    selected_types = set(selected_source_types(args))
    selected_doc_ids = set(args.doc_id)
    in_scope = [
        source
        for source in sources
        if source_matches_request(
            source,
            selected_types=selected_types,
            selected_doc_ids=selected_doc_ids,
            doc_version=args.doc_version,
            include_all_sources=args.include_all_sources,
        )
    ]
    ready_in_scope = [source for source in in_scope if source.get("status") == "ready"]
    warnings = source_inventory_warnings(
        sources=sources,
        in_scope=in_scope,
        ready_in_scope=ready_in_scope,
        doc_version=args.doc_version,
        selected_doc_ids=selected_doc_ids,
        include_all_sources=args.include_all_sources,
    )
    identifiers = (
        source_identifiers(sources, limit=args.source_identifier_limit)
        if args.include_source_identifiers
        else []
    )
    return {
        "ok": True,
        "error": "",
        "warnings": warnings,
        "total_sources": len(sources),
        "in_scope_sources": len(in_scope),
        "ready_in_scope_sources": len(ready_in_scope),
        "total_chunks": sum_int(source.get("chunk_count") for source in sources),
        "in_scope_chunks": sum_int(source.get("chunk_count") for source in in_scope),
        "ready_in_scope_chunks": sum_int(source.get("chunk_count") for source in ready_in_scope),
        "by_type": count_by(sources, "source_type"),
        "in_scope_by_type": count_by(in_scope, "source_type"),
        "by_status": count_by(sources, "status"),
        "in_scope_by_status": count_by(in_scope, "status"),
        "current_sources": sum(1 for source in sources if source.get("current")),
        "current_in_scope_sources": sum(1 for source in in_scope if source.get("current")),
        "filters": {
            "source_types": sorted(selected_types),
            "doc_ids": len(selected_doc_ids),
            "doc_version": args.doc_version,
            "include_all_sources": args.include_all_sources,
        },
        "identifiers_included": bool(args.include_source_identifiers),
        "source_identifier_limit": args.source_identifier_limit,
        "source_identifiers": identifiers,
    }


def source_inventory_warnings(
    *,
    sources: list[dict[str, Any]],
    in_scope: list[dict[str, Any]],
    ready_in_scope: list[dict[str, Any]],
    doc_version: int | None,
    selected_doc_ids: set[str],
    include_all_sources: bool,
) -> list[str]:
    warnings: list[str] = []
    current_sources = [source for source in sources if source.get("current")]
    if not include_all_sources and sources and not current_sources and doc_version is None:
        warnings.append(
            "current_sources=0: --current-only was set, but no visible document is current."
        )
    if sources and not in_scope:
        if selected_doc_ids or doc_version is not None:
            warnings.append(
                "in_scope_sources=0: the selected --doc-id/--doc-version/source-type filters match no visible source for this token."
            )
        else:
            warnings.append(
                "in_scope_sources=0: no visible source matches the default source-type filters. "
                "Run with --include-source-identifiers to inspect doc_id/doc_version."
            )
    elif in_scope and not ready_in_scope:
        warnings.append(
            "ready_in_scope_sources=0: matching sources exist but none are ready, so retrieval pressure testing will not exercise ready document chunks."
        )
    return warnings


def source_identifiers(sources: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    safe_limit = max(0, limit)
    identifiers: list[dict[str, Any]] = []
    for source in sources[:safe_limit]:
        identifiers.append(
            {
                "doc_id": str(source.get("doc_id") or ""),
                "doc_version": source.get("doc_version"),
                "source_type": str(source.get("source_type") or ""),
                "status": str(source.get("status") or ""),
                "current": bool(source.get("current")),
                "chunk_count": int(source.get("chunk_count") or 0),
            }
        )
    return identifiers


def enforce_rag_inventory(inventory: dict[str, Any]) -> None:
    if not inventory.get("ok"):
        raise SystemExit(
            "RAG load test aborted: could not verify /sources inventory. "
            f"Reason: {inventory.get('error') or 'unknown error'}"
        )
    if int(inventory.get("in_scope_sources") or 0) <= 0:
        raise SystemExit(
            "RAG load test aborted: in_scope_sources=0, so this run would not exercise RAG retrieval. "
            "Check that this token can see at least one ready source."
        )
    if int(inventory.get("ready_in_scope_sources") or 0) <= 0:
        raise SystemExit(
            "RAG load test aborted: ready_in_scope_sources=0, so no ready document can be retrieved. "
            "Wait for ingestion to finish or choose filters that include at least one ready source."
        )


def source_matches_request(
    source: dict[str, Any],
    *,
    selected_types: set[str],
    selected_doc_ids: set[str],
    doc_version: int | None,
    include_all_sources: bool,
) -> bool:
    if not source_matches_filters(
        source,
        selected_types=selected_types,
        selected_doc_ids=selected_doc_ids,
    ):
        return False
    if doc_version is not None and source.get("doc_version") != doc_version:
        return False
    if not include_all_sources and doc_version is None and not source.get("current", False):
        return False
    return True


def source_matches_filters(
    source: dict[str, Any],
    *,
    selected_types: set[str],
    selected_doc_ids: set[str],
) -> bool:
    if selected_types and source.get("source_type") not in selected_types:
        return False
    if selected_doc_ids and source.get("doc_id") not in selected_doc_ids:
        return False
    return True


def count_by(sources: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(str(source.get(field) or "unknown") for source in sources)
    return dict(sorted(counter.items()))


def sum_int(values) -> int:
    total = 0
    for value in values:
        try:
            total += int(value or 0)
        except (TypeError, ValueError):
            continue
    return total


def mem_total_mb() -> int | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024)
    return None


def selected_source_types(args: argparse.Namespace) -> list[str]:
    return args.source_type or DEFAULT_SOURCE_TYPES


def run_and_print_stage(args: argparse.Namespace, *, output_dir: Path, concurrency: int) -> StageResult:
    result = run_stage(args, output_dir=output_dir, concurrency=concurrency)
    status = "usable" if result.usable else "unusable"
    print(
        f"concurrency={concurrency} {status} "
        f"success={result.summary['success']}/{result.summary['requests']} "
        f"p95={result.summary['latency_ms']['p95']}ms "
        f"rps={result.summary['throughput_rps']} "
        f"reasons={'; '.join(result.reasons) or '-'}",
        flush=True,
    )
    return result


def run_search_limit(args: argparse.Namespace, *, output_dir: Path) -> list[StageResult]:
    low = max(1, int(args.search_min_concurrency))
    high_limit = max(low, int(args.search_max_concurrency))
    precision = max(1, int(args.search_precision))
    results: list[StageResult] = []
    tested: dict[int, StageResult] = {}

    def run_once(concurrency: int) -> StageResult:
        if concurrency in tested:
            return tested[concurrency]
        result = run_and_print_stage(args, output_dir=output_dir, concurrency=concurrency)
        tested[concurrency] = result
        results.append(result)
        return result

    print("Search phase: exponential probe", flush=True)
    first = run_once(low)
    if not first.usable:
        return results

    best_good = low
    first_bad: int | None = None
    current = low
    while current < high_limit:
        current = min(high_limit, current * 2)
        result = run_once(current)
        if result.usable:
            best_good = current
            if current == high_limit:
                return results
            continue
        first_bad = current
        break

    if first_bad is None:
        return results

    print(
        f"Search phase: binary search between usable={best_good} and unusable={first_bad}",
        flush=True,
    )
    while first_bad - best_good > precision:
        midpoint = (best_good + first_bad) // 2
        if midpoint in (best_good, first_bad):
            break
        result = run_once(midpoint)
        if result.usable:
            best_good = midpoint
        else:
            first_bad = midpoint
    print(
        f"Approximate concurrency limit: highest usable={best_good}, first unusable={first_bad}",
        flush=True,
    )
    return results


def run_stage(args: argparse.Namespace, *, output_dir: Path, concurrency: int) -> StageResult:
    requests = request_count(args, concurrency)
    json_path = output_dir / f"concurrency-{concurrency}.json"
    stdout_path = output_dir / f"concurrency-{concurrency}.stdout"
    stderr_path = output_dir / f"concurrency-{concurrency}.stderr"
    command = stage_command(
        args,
        concurrency=concurrency,
        requests=requests,
        json_path=json_path,
    )
    print(f"Running: {' '.join(shlex.quote(part) for part in command)}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        summary = {
            "requests": requests,
            "concurrency": concurrency,
            "success": 0,
            "failed": requests,
            "throughput_rps": 0.0,
            "latency_ms": empty_summary(),
            "first_event_ms": empty_summary(),
            "external_latency_ms": {},
            "stage_latency_ms": {},
            "citations": empty_summary(),
            "failures": [{"error": f"runner exited with code {completed.returncode}"}],
        }
    else:
        summary = json.loads(json_path.read_text(encoding="utf-8"))
    usable, reasons = evaluate_stage(args, summary)
    return StageResult(
        concurrency=concurrency,
        summary=summary,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        json_path=json_path,
        usable=usable,
        reasons=reasons,
    )


def stage_command(
    args: argparse.Namespace,
    *,
    concurrency: int,
    requests: int,
    json_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--base-url",
        args.base_url,
        "--endpoint",
        args.endpoint,
        "--external-mode",
        "real",
        "--concurrency",
        str(concurrency),
        "--requests",
        str(requests),
        "--warmup",
        str(args.warmup),
        "--timeout",
        str(args.timeout),
        "--candidate-limit",
        str(args.candidate_limit),
        "--context-limit",
        str(args.context_limit),
        "--tenant-id",
        args.tenant_id,
        "--question",
        args.question,
        "--json-output",
        str(json_path),
    ]
    if args.token:
        command.extend(["--token", args.token])
    if args.questions_file:
        command.extend(["--questions-file", str(args.questions_file)])
    if args.doc_version is not None:
        command.extend(["--doc-version", str(args.doc_version)])
    if args.include_all_sources:
        command.append("--include-all-sources")
    for source_type in selected_source_types(args):
        command.extend(["--source-type", source_type])
    for doc_id in args.doc_id:
        command.extend(["--doc-id", doc_id])
    for group in args.acl_group:
        command.extend(["--acl-group", group])
    return command


def evaluate_stage(args: argparse.Namespace, summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    requests = max(1, int(summary.get("requests") or 0))
    failed = int(summary.get("failed") or 0)
    failure_rate = failed / requests
    p95 = float((summary.get("latency_ms") or {}).get("p95") or 0)
    first_event_p95 = float((summary.get("first_event_ms") or {}).get("p95") or 0)
    throughput = float(summary.get("throughput_rps") or 0)
    citation_avg = float((summary.get("citations") or {}).get("avg") or 0)
    if failure_rate > args.max_failure_rate:
        reasons.append(f"failure_rate={failure_rate:.2%} > {args.max_failure_rate:.2%}")
    if p95 <= 0 or p95 > args.max_p95_ms:
        reasons.append(f"p95_ms={p95:.2f} > {args.max_p95_ms:.2f}")
    if first_event_p95 <= 0 or first_event_p95 > args.max_first_event_p95_ms:
        reasons.append(
            f"first_event_p95_ms={first_event_p95:.2f} > {args.max_first_event_p95_ms:.2f}"
        )
    if throughput < args.min_throughput_rps:
        reasons.append(f"throughput_rps={throughput:.3f} < {args.min_throughput_rps:.3f}")
    if not args.allow_zero_citations and citation_avg <= 0:
        reasons.append("citations_avg=0; RAG retrieval may not be exercised")
    return not reasons, reasons


def empty_summary() -> dict[str, float]:
    return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}


def render_report(
    *,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    health: dict[str, Any],
    results: list[StageResult],
    output_dir: Path,
) -> str:
    best = best_usable(results)
    first_bad = next((result for result in results if not result.usable), None)
    lines = [
        "# Production RAG Deployed Server Load Test",
        "",
        f"Date: {metadata['started_at']}",
        "",
        "## Target",
        "",
        f"- Server profile: `{metadata['server_profile']}`",
        f"- Base URL: `{metadata['base_url']}`",
        f"- Endpoint: `{metadata['endpoint']}`",
        f"- Health: `{json.dumps(health, ensure_ascii=False)}`",
        f"- Host CPU count seen by script: `{metadata['host']['cpu_count']}`",
        f"- Host memory seen by script: `{metadata['host']['mem_total_mb']} MB`",
        "",
        "## Version",
        "",
    ]
    version = metadata["version"]
    for key in ("api_version", "local_version", "git_branch", "git_commit", "worktree"):
        if key in version:
            lines.append(f"- {key}: `{version[key]}`")
    lines.extend(
        [
            "",
            "## Method",
            "",
            method_description(metadata),
            "A level is considered usable only when all configured thresholds pass.",
            "",
            "Thresholds:",
            "",
            f"- Max failure rate: `{metadata['thresholds']['max_failure_rate']:.2%}`",
            f"- Max P95 latency: `{metadata['thresholds']['max_p95_ms']} ms`",
            f"- Max first event P95: `{metadata['thresholds']['max_first_event_p95_ms']} ms`",
            f"- Min throughput: `{metadata['thresholds']['min_throughput_rps']} rps`",
            f"- Require citations: `{metadata['thresholds']['require_citations']}`",
            "",
            "Request settings:",
            "",
            f"- Question: `{metadata['request']['question']}`",
            f"- Source types: `{', '.join(metadata['request']['source_types'])}`",
            f"- Doc IDs: `{', '.join(metadata['request']['doc_ids']) or '-'}`",
            f"- Doc version: `{metadata['request']['doc_version']}`",
            f"- Candidate limit: `{metadata['request']['candidate_limit']}`",
            f"- Context limit: `{metadata['request']['context_limit']}`",
            "",
        ]
    )
    if metadata.get("notes"):
        lines.extend(["Notes:", "", metadata["notes"], ""])
    inventory = metadata.get("source_inventory") or {}
    lines.extend(render_inventory_section(inventory))
    lines.extend(
        [
            "## Result Summary",
            "",
            "| Concurrency | Requests | Success | Failed | Usable | Avg latency | P95 latency | First event P95 | Throughput | Citations avg | Reasons |",
            "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        summary = result.summary
        lines.append(
            "| "
            f"{result.concurrency} | "
            f"{summary['requests']} | "
            f"{summary['success']} | "
            f"{summary['failed']} | "
            f"{'yes' if result.usable else 'no'} | "
            f"{summary['latency_ms']['avg']} ms | "
            f"{summary['latency_ms']['p95']} ms | "
            f"{summary['first_event_ms']['p95']} ms | "
            f"{summary['throughput_rps']} rps | "
            f"{summary['citations']['avg']} | "
            f"{'<br>'.join(result.reasons) or '-'} |"
        )
    lines.extend(["", "## Stage Averages", ""])
    lines.extend(
        [
            "| Concurrency | LLM | Embedding | Rerank | Milvus/Search | Unattributed |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        summary = result.summary
        external = summary.get("external_latency_ms") or {}
        stage = summary.get("stage_latency_ms") or {}
        search = stage.get("milvus_search") or stage.get("search") or {}
        lines.append(
            "| "
            f"{result.concurrency} | "
            f"{metric(external, 'llm')} ms | "
            f"{metric(external, 'embedding')} ms | "
            f"{metric(external, 'rerank')} ms | "
            f"{float(search.get('avg') or 0):.2f} ms | "
            f"{summary['unattributed_ms']['avg']} ms |"
        )
    lines.extend(["", "## Conclusion", ""])
    if best:
        lines.extend(
            [
                f"- Highest usable concurrency: `{best.concurrency}`",
                f"- At that level: `{best.summary['throughput_rps']} rps`, P95 `{best.summary['latency_ms']['p95']} ms`, failures `{best.summary['failed']}`",
            ]
        )
    else:
        lines.append("- No configured concurrency level met the usable thresholds.")
    if first_bad:
        lines.append(
            f"- First unusable concurrency: `{first_bad.concurrency}` because {'; '.join(first_bad.reasons)}"
        )
    else:
        lines.append("- No unusable level was found in the configured range; increase `--concurrency-levels` to push harder.")
    lines.extend(
        [
            "",
            "## Raw Files",
            "",
            f"- Output directory: `{output_dir}`",
            "- Each level has `.json`, `.stdout`, and `.stderr` files for audit.",
            "",
            "## Reproduce",
            "",
            "Run the same script with the same options from the repository root. Example:",
            "",
            "```bash",
            "source .venv/bin/activate",
            "export RAG_LOAD_TEST_TOKEN='your-token-here'",
            "python projects/09-production-rag/tests/load/production_limit_load.py \\",
            f"  --base-url {shlex.quote(args.base_url)} \\",
            f"  --question {shlex.quote(args.question)}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_inventory_section(inventory: dict[str, Any]) -> list[str]:
    lines = ["## Data Inventory", ""]
    if not inventory.get("ok"):
        lines.extend(
            [
                "The script could not fetch `/sources` for this token.",
                "",
                f"- Error: `{inventory.get('error') or 'unknown'}`",
                "",
            ]
        )
        return lines
    lines.extend(
        [
            "The source inventory was fetched with the same token used for the pressure test.",
            "Only counts are recorded; source titles and filenames are intentionally omitted.",
            "",
            "| Metric | Count |",
            "| --- | ---: |",
            f"| Total sources visible to token | {inventory['total_sources']} |",
            f"| Total chunks visible to token | {inventory['total_chunks']} |",
            f"| Current sources visible to token | {inventory['current_sources']} |",
            f"| Sources matching this test's filters | {inventory['in_scope_sources']} |",
            f"| Ready sources matching filters | {inventory['ready_in_scope_sources']} |",
            f"| Chunks matching filters | {inventory['in_scope_chunks']} |",
            f"| Ready chunks matching filters | {inventory['ready_in_scope_chunks']} |",
            "",
        ]
    )
    warnings = inventory.get("warnings") or []
    if warnings:
        lines.extend(["Warnings:", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(
        [
            "By source type:",
            "",
            "| Source type | Visible sources | In-scope sources |",
            "| --- | ---: | ---: |",
        ]
    )
    all_types = sorted(
        set((inventory.get("by_type") or {}).keys())
        | set((inventory.get("in_scope_by_type") or {}).keys())
    )
    if not all_types:
        lines.append("| - | 0 | 0 |")
    for source_type in all_types:
        lines.append(
            f"| {source_type} | "
            f"{(inventory.get('by_type') or {}).get(source_type, 0)} | "
            f"{(inventory.get('in_scope_by_type') or {}).get(source_type, 0)} |"
        )
    lines.extend(
        [
            "",
            "By source status:",
            "",
            "| Status | Visible sources | In-scope sources |",
            "| --- | ---: | ---: |",
        ]
    )
    all_statuses = sorted(
        set((inventory.get("by_status") or {}).keys())
        | set((inventory.get("in_scope_by_status") or {}).keys())
    )
    if not all_statuses:
        lines.append("| - | 0 | 0 |")
    for status in all_statuses:
        lines.append(
            f"| {status} | "
            f"{(inventory.get('by_status') or {}).get(status, 0)} | "
            f"{(inventory.get('in_scope_by_status') or {}).get(status, 0)} |"
        )
    filters = inventory.get("filters") or {}
    lines.extend(
        [
            "",
            "Inventory filters:",
            "",
            f"- Source types: `{', '.join(filters.get('source_types') or [])}`",
            f"- Explicit doc id count: `{filters.get('doc_ids', 0)}`",
            f"- Doc version: `{filters.get('doc_version')}`",
            f"- Include all visible sources: `{filters.get('include_all_sources')}`",
            "",
        ]
    )
    identifiers = inventory.get("source_identifiers") or []
    if identifiers:
        lines.extend(
            [
                "Source identifiers:",
                "",
                "| doc_id | doc_version | source_type | status | current | chunks |",
                "| --- | ---: | --- | --- | --- | ---: |",
            ]
        )
        for item in identifiers:
            lines.append(
                "| "
                f"`{markdown_cell(item.get('doc_id') or '')}` | "
                f"{item.get('doc_version')} | "
                f"{markdown_cell(item.get('source_type') or '')} | "
                f"{markdown_cell(item.get('status') or '')} | "
                f"{item.get('current')} | "
                f"{item.get('chunk_count')} |"
            )
        limit = inventory.get("source_identifier_limit")
        total = inventory.get("total_sources")
        lines.extend(
            [
                "",
                f"Only the first `{limit}` of `{total}` visible sources are listed.",
                "",
            ]
        )
    elif inventory.get("identifiers_included"):
        lines.extend(["Source identifiers were requested, but no visible sources were returned.", ""])
    else:
        lines.extend(
            [
                "Doc identifiers are omitted by default. Re-run with `--include-source-identifiers` to include a limited doc_id/doc_version table in the report.",
                "",
            ]
        )
    return lines


def method_description(metadata: dict[str, Any]) -> str:
    search = metadata.get("search") or {}
    if search.get("enabled"):
        return (
            "This test first probes concurrency exponentially, then binary-searches the "
            f"usable/unusable boundary up to `{search.get('max_concurrency')}` "
            f"with precision `{search.get('precision')}`."
        )
    return "This test ramps concurrency through the configured levels."


def markdown_cell(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def metric(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key) or {}
    return float(value.get("avg") or 0)


def best_usable(results: list[StageResult]) -> StageResult | None:
    usable = [result for result in results if result.usable]
    return usable[-1] if usable else None


def best_usable_payload(results: list[StageResult]) -> dict[str, Any] | None:
    best = best_usable(results)
    if best is None:
        return None
    return {
        "concurrency": best.concurrency,
        "throughput_rps": best.summary.get("throughput_rps"),
        "p95_ms": (best.summary.get("latency_ms") or {}).get("p95"),
        "failed": best.summary.get("failed"),
        "requests": best.summary.get("requests"),
    }


def first_unusable(results: list[StageResult]) -> StageResult | None:
    return next((result for result in sorted(results, key=lambda item: item.concurrency) if not result.usable), None)


def limit_summary_payload(results: list[StageResult]) -> dict[str, Any]:
    best = best_usable(results)
    bad = first_unusable(results)
    return {
        "highest_usable_concurrency": best.concurrency if best else None,
        "first_unusable_concurrency": bad.concurrency if bad else None,
        "limit_reached": bad is not None,
        "highest_tested_concurrency": max((result.concurrency for result in results), default=None),
    }


def print_final_summary(results: list[StageResult], *, search_limit: bool) -> None:
    best = best_usable(results)
    bad = first_unusable(results)
    print("Final load-test result:", flush=True)
    if best is None:
        print("- Highest usable concurrency: none", flush=True)
        if bad is not None:
            print(f"- First unusable concurrency: {bad.concurrency}", flush=True)
            print(f"- Reasons: {'; '.join(bad.reasons) or '-'}", flush=True)
        return

    print(f"- Highest usable concurrency: {best.concurrency}", flush=True)
    print(f"- At that level: requests={best.summary.get('requests')}, success={best.summary.get('success')}, failed={best.summary.get('failed')}", flush=True)
    print(f"- P95 latency: {(best.summary.get('latency_ms') or {}).get('p95')} ms", flush=True)
    print(f"- Throughput: {best.summary.get('throughput_rps')} rps", flush=True)
    print(f"- Citations avg: {(best.summary.get('citations') or {}).get('avg')}", flush=True)
    if bad is not None:
        print(f"- First unusable concurrency: {bad.concurrency}", flush=True)
        print(f"- Approximate limit: {best.concurrency} usable, {bad.concurrency} unusable", flush=True)
    elif search_limit:
        print("- First unusable concurrency: not found in configured search range", flush=True)
        print(f"- Approximate limit: >= {best.concurrency}", flush=True)


def print_dry_run(args: argparse.Namespace, *, output_dir: Path, levels: list[int]) -> None:
    print(f"Output directory: {output_dir}", flush=True)
    if args.search_limit:
        print(
            "Adaptive search mode: exponential probing followed by binary search. "
            "Binary-search commands depend on previous pass/fail results.",
            flush=True,
        )
        levels = exponential_probe_preview(
            min_concurrency=args.search_min_concurrency,
            max_concurrency=args.search_max_concurrency,
        )
    for level in levels:
        requests = request_count(args, level)
        command = stage_command(args, concurrency=level, requests=requests, json_path=output_dir / f"concurrency-{level}.json")
        print(" ".join(shlex.quote(part) for part in command), flush=True)


def request_count(args: argparse.Namespace, concurrency: int) -> int:
    requests = max(args.min_requests, concurrency * args.requests_per_concurrency)
    if args.max_requests and args.max_requests > 0:
        return min(args.max_requests, requests)
    return requests


def exponential_probe_preview(*, min_concurrency: int, max_concurrency: int) -> list[int]:
    low = max(1, int(min_concurrency))
    high = max(low, int(max_concurrency))
    levels = [low]
    current = low
    while current < high:
        current = min(high, current * 2)
        levels.append(current)
    return levels


def run_command(command: list[str], *, cwd: Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


if __name__ == "__main__":
    main()
