from __future__ import annotations

from fastapi.testclient import TestClient

from rag_core.app_version import app_version
from serve import create_app


def main() -> None:
    expected = app_version()
    assert expected != "0.0.0"
    api = TestClient(create_app())
    openapi = api.get("/openapi.json")
    assert openapi.status_code == 200, openapi.text
    assert openapi.json()["info"]["version"] == expected
    print("smoke_app_version=ok")


if __name__ == "__main__":
    main()
