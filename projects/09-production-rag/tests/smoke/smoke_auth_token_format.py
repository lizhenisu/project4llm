from __future__ import annotations

import re

from rag_core.user_auth import generate_session_token


def main() -> None:
    tokens = {generate_session_token() for _ in range(32)}
    assert len(tokens) == 32
    assert all(re.fullmatch(r"[a-z0-9]{24}", token) for token in tokens)
    print("smoke_auth_token_format=ok")


if __name__ == "__main__":
    main()
