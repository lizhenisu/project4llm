from __future__ import annotations

import os

import rag_core.config as config_module
from rag_core.config import load_config
from rag_core.readiness import redacted_config


LEGACY_BACKENDS = {
    "RAG_ANSWER_BACKEND": "extractive",
    "RAG_QUERY_REWRITE_BACKEND": "none",
    "RAG_EMBEDDING_BACKEND": "hash",
    "RAG_RERANK_BACKEND": "none",
}


def main() -> None:
    old_env = {name: os.environ.get(name) for name in LEGACY_BACKENDS}
    old_siliconflow_key = os.environ.get("SILICONFLOW_API_KEY")
    try:
        for name, value in LEGACY_BACKENDS.items():
            reset_env(old_env)
            os.environ[name] = value
            config_module._ENV_LOADED = True
            try:
                load_config()
            except ValueError as exc:
                assert name in str(exc)
            else:
                raise AssertionError(f"{name}={value} should be rejected")

        reset_env(old_env)
        os.environ["SILICONFLOW_API_KEY"] = "secret-siliconflow-key"
        config_module._ENV_LOADED = True
        assert redacted_config(load_config())["siliconflow_api_key"] == "***"
    finally:
        reset_env(old_env)
        if old_siliconflow_key is None:
            os.environ.pop("SILICONFLOW_API_KEY", None)
        else:
            os.environ["SILICONFLOW_API_KEY"] = old_siliconflow_key
        config_module._ENV_LOADED = False

    print("smoke_backend_config_guards=ok")


def reset_env(values: dict[str, str | None]) -> None:
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


if __name__ == "__main__":
    main()
