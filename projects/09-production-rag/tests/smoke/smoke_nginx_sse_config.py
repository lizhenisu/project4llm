from __future__ import annotations

import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
NGINX_CONFIG = PROJECT_DIR / "frontend" / "nginx.conf"


def main() -> None:
    config = NGINX_CONFIG.read_text(encoding="utf-8")
    upstream = upstream_block(config, "rag_api_upstream")
    stream_location = location_block(config, "= /api/query/stream")
    api_location = location_block(config, "/api/")

    assert "zone rag_api_upstream 64k;" in upstream
    assert "resolver 127.0.0.11 valid=5s ipv6=off;" in upstream
    assert "resolver_timeout 2s;" in upstream
    assert "server rag-api:8008 resolve;" in upstream
    assert "keepalive 64;" in upstream

    assert "proxy_pass http://rag_api_upstream/query/stream;" in stream_location
    assert "proxy_http_version 1.1;" in stream_location
    assert 'proxy_set_header Connection "";' in stream_location
    assert "proxy_buffering off;" in stream_location
    assert "proxy_cache off;" in stream_location
    assert "gzip off;" in stream_location
    assert "proxy_read_timeout 600s;" in stream_location
    assert "send_timeout 600s;" in stream_location
    assert "add_header X-Accel-Buffering no always;" in stream_location

    assert "proxy_pass http://rag_api_upstream/;" in api_location
    assert "proxy_http_version 1.1;" in api_location
    assert 'proxy_set_header Connection "";' in api_location
    assert "proxy_next_upstream_tries 2;" in api_location
    for location in (stream_location, api_location):
        assert "proxy_set_header X-Forwarded-Host $host;" in location
        assert "proxy_set_header X-Forwarded-Port $server_port;" in location
    print("smoke_nginx_sse_config=ok")


def location_block(config: str, location: str) -> str:
    match = re.search(
        rf"location\s+{re.escape(location)}\s*\{{(?P<body>.*?)^\s*\}}",
        config,
        flags=re.DOTALL | re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"missing nginx location: {location}")
    return match.group("body")


def upstream_block(config: str, name: str) -> str:
    match = re.search(
        rf"upstream\s+{re.escape(name)}\s*\{{(?P<body>.*?)^\s*\}}",
        config,
        flags=re.DOTALL | re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"missing nginx upstream: {name}")
    return match.group("body")


if __name__ == "__main__":
    main()
