from __future__ import annotations

import json
import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, unquote

from rag_core.io import read_jsonl, write_jsonl


def object_store_backend() -> str:
    return os.environ.get("RAG_OBJECT_STORE_BACKEND", "local").lower()


def read_object_jsonl(object_store_dir: Path, relative_path: Path) -> list[dict]:
    if object_store_backend() == "s3":
        body = read_s3_text(relative_path)
        if not body:
            return []
        rows: list[dict] = []
        for line_no, line in enumerate(body.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at s3://{s3_bucket()}/{s3_key(relative_path)}:{line_no}") from exc
        return rows
    path = object_store_dir / relative_path
    return read_jsonl(path) if path.exists() else []


def write_object_jsonl(object_store_dir: Path, relative_path: Path, rows: Iterable[dict]) -> None:
    if object_store_backend() == "s3":
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        write_s3_text(relative_path, text)
        return
    path = object_store_dir / relative_path
    write_jsonl(path, rows)


def object_exists(object_store_dir: Path, relative_path: Path) -> bool:
    if object_store_backend() == "s3":
        client = s3_client()
        try:
            client.head_object(Bucket=s3_bucket(), Key=s3_key(relative_path))
            return True
        except Exception:
            return False
    return (object_store_dir / relative_path).exists()


def delete_object(object_store_dir: Path, relative_path: Path) -> bool:
    if object_store_backend() == "s3":
        client = s3_client()
        key = s3_key(relative_path)
        existed = object_exists(object_store_dir, relative_path)
        client.delete_object(Bucket=s3_bucket(), Key=key)
        return existed
    path = object_store_dir / relative_path
    if not path.exists():
        return False
    path.unlink()
    return True


def upload_file_to_object_store(local_path: Path, relative_path: Path, *, content_type: str | None = None) -> str:
    if object_store_backend() == "s3":
        ensure_s3_bucket()
        extra_args = {"ContentType": content_type} if content_type else {}
        key = s3_key(relative_path)
        s3_client().upload_file(str(local_path), s3_bucket(), key, ExtraArgs=extra_args)
        return s3_uri_for_key(key)
    return str(local_path)


def read_object_bytes_by_uri(uri: str) -> bytes:
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        response = s3_client().get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    return Path(uri).read_bytes()


def read_object_bytes_by_relative_path(object_store_dir: Path, relative_path: Path) -> bytes:
    if object_store_backend() == "s3":
        response = s3_client().get_object(Bucket=s3_bucket(), Key=s3_key(relative_path))
        return response["Body"].read()
    return (object_store_dir / relative_path).read_bytes()


def object_uri_for_relative_path(relative_path: Path) -> str:
    if object_store_backend() == "s3":
        return s3_uri_for_key(s3_key(relative_path))
    return relative_path.as_posix()


def s3_uri_for_key(key: str) -> str:
    return f"s3://{s3_bucket()}/{key}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    path = uri[len("s3://") :]
    bucket, _, key = path.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def quote_object_uri(uri: str) -> str:
    return quote(uri, safe="")


def unquote_object_uri(value: str) -> str:
    return unquote(value)


def read_s3_text(relative_path: Path) -> str:
    client = s3_client()
    try:
        response = client.get_object(Bucket=s3_bucket(), Key=s3_key(relative_path))
    except client.exceptions.NoSuchKey:
        return ""
    except Exception as exc:
        if exc.__class__.__name__ == "ClientError" and exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return ""
        raise
    return response["Body"].read().decode("utf-8")


def write_s3_text(relative_path: Path, text: str) -> None:
    ensure_s3_bucket()
    s3_client().put_object(
        Bucket=s3_bucket(),
        Key=s3_key(relative_path),
        Body=text.encode("utf-8"),
        ContentType="application/x-ndjson; charset=utf-8",
    )


@lru_cache(maxsize=1)
def s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("RAG_S3_ENDPOINT_URL") or None,
        aws_access_key_id=os.environ.get("RAG_S3_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.environ.get("RAG_S3_SECRET_ACCESS_KEY") or None,
        region_name=os.environ.get("RAG_S3_REGION", "us-east-1"),
    )


def ensure_s3_bucket() -> None:
    client = s3_client()
    bucket = s3_bucket()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)


def s3_bucket() -> str:
    return os.environ.get("RAG_S3_BUCKET", "production-rag")


def s3_prefix() -> str:
    return os.environ.get("RAG_S3_PREFIX", "").strip("/")


def s3_key(relative_path: Path) -> str:
    path = relative_path.as_posix().lstrip("/")
    prefix = s3_prefix()
    return f"{prefix}/{path}" if prefix else path
