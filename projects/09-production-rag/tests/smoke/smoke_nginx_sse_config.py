from __future__ import annotations

import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
NGINX_CONFIG = PROJECT_DIR / "frontend" / "nginx.conf"


def main() -> None:
    config = NGINX_CONFIG.read_text(encoding="utf-8")
    stream_location = location_block(config, "= /api/query/stream")
    api_location = location_block(config, "/api/")

    assert "proxy_pass http://rag-api:8008/query/stream;" in stream_location
    assert "proxy_http_version 1.1;" in stream_location
    assert 'proxy_set_header Connection "";' in stream_location
    assert "proxy_buffering off;" in stream_location
    assert "proxy_cache off;" in stream_location
    assert "gzip off;" in stream_location
    assert "proxy_read_timeout 600s;" in stream_location
    assert "send_timeout 600s;" in stream_location
    assert "add_header X-Accel-Buffering no always;" in stream_location

    assert "proxy_http_version 1.1;" in api_location
    assert 'proxy_set_header Connection "";' in api_location
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


if __name__ == "__main__":
    main()
