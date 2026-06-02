from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

try:
    from pymilvus import DataType, MilvusClient
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pymilvus. Run `pip install pymilvus` inside the activated venv."
    ) from exc


DIM = 64
COLLECTION_NAME = "rag_chunks_demo"
DB_PATH = Path(__file__).with_name("milvus_lite_demo.db")
TENANT_ID_MAX_LENGTH = 32
SOURCE_MAX_LENGTH = 32
DOC_ID_MAX_LENGTH = 64
TEXT_MAX_LENGTH = 512


DOCUMENTS = [
    {
        "id": 1,
        "tenant_id": "team_a",
        "source": "handbook",
        "doc_id": "agent-rag-001",
        "text": "RAG 系统通常先进行 query rewrite，再检索 topK chunk，最后把证据片段拼入 prompt。",
    },
    {
        "id": 2,
        "tenant_id": "team_a",
        "source": "handbook",
        "doc_id": "milvus-001",
        "text": "Milvus collection 保存向量字段和标量 metadata，检索时可以结合 tenant_id 做过滤。",
    },
    {
        "id": 3,
        "tenant_id": "team_a",
        "source": "runbook",
        "doc_id": "milvus-ops-001",
        "text": "如果向量检索延迟过高，应检查 topK、索引参数、过滤字段索引以及 reranker 耗时。",
    },
    {
        "id": 4,
        "tenant_id": "team_b",
        "source": "handbook",
        "doc_id": "finance-001",
        "text": "财务知识库的报销规则只允许 team_b 成员检索，其他租户不能看到这些 chunk。",
    },
    {
        "id": 5,
        "tenant_id": "team_a",
        "source": "handbook",
        "doc_id": "index-001",
        "text": "IVF 索引通过聚类缩小搜索范围，nprobe 越大召回通常越高但查询延迟也越高。",
    },
    {
        "id": 6,
        "tenant_id": "team_a",
        "source": "handbook",
        "doc_id": "hnsw-001",
        "text": "HNSW 使用近邻图做 ANN 搜索，ef 越大通常召回越高，内存开销也需要重点评估。",
    },
]


EVAL_QUERIES = [
    {
        "query": "Milvus 如何用 metadata 做租户过滤？",
        "tenant_id": "team_a",
        "relevant_doc_ids": {"milvus-001"},
    },
    {
        "query": "向量检索变慢应该排查哪些因素？",
        "tenant_id": "team_a",
        "relevant_doc_ids": {"milvus-ops-001"},
    },
    {
        "query": "IVF 的 nprobe 会影响什么？",
        "tenant_id": "team_a",
        "relevant_doc_ids": {"index-001"},
    },
]


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    tokens: list[str] = []
    current: list[str] = []
    for char in normalized:
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def hash_embedding(text: str, dim: int = DIM) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def build_entities() -> list[dict]:
    return [
        {
            **doc,
            "vector": hash_embedding(
                f"{doc['source']} {doc['doc_id']} {doc['tenant_id']} {doc['text']}"
            ),
        }
        for doc in DOCUMENTS
    ]


def reset_collection(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=DIM,
    )
    schema.add_field(
        field_name="tenant_id",
        datatype=DataType.VARCHAR,
        max_length=TENANT_ID_MAX_LENGTH,
    )
    schema.add_field(
        field_name="source",
        datatype=DataType.VARCHAR,
        max_length=SOURCE_MAX_LENGTH,
    )
    schema.add_field(
        field_name="doc_id",
        datatype=DataType.VARCHAR,
        max_length=DOC_ID_MAX_LENGTH,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=TEXT_MAX_LENGTH,
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        metric_type="COSINE",
    )


def print_hits(title: str, hits: Iterable[dict]) -> None:
    print(f"\n{title}")
    for rank, hit in enumerate(hits, start=1):
        entity = hit["entity"]
        print(
            f"{rank}. score={hit['distance']:.4f} "
            f"tenant={entity['tenant_id']} source={entity['source']} "
            f"doc={entity['doc_id']} text={entity['text']}"
        )


def search(
    client: MilvusClient,
    query: str,
    *,
    limit: int = 3,
    filter_expr: str = "",
) -> list[dict]:
    result = client.search(
        collection_name=COLLECTION_NAME,
        data=[hash_embedding(query)],
        limit=limit,
        filter=filter_expr,
        output_fields=["text", "doc_id", "tenant_id", "source"],
    )
    return result[0]


def recall_at_k(client: MilvusClient, k: int = 3) -> float:
    hits = 0
    for item in EVAL_QUERIES:
        results = search(
            client,
            item["query"],
            limit=k,
            filter_expr=f"tenant_id == '{item['tenant_id']}'",
        )
        returned = {hit["entity"]["doc_id"] for hit in results}
        if returned & item["relevant_doc_ids"]:
            hits += 1
    return hits / len(EVAL_QUERIES)


def main() -> None:
    client = MilvusClient(str(DB_PATH))
    reset_collection(client)

    entities = build_entities()
    insert_result = client.insert(collection_name=COLLECTION_NAME, data=entities)
    print(f"Inserted rows: {insert_result['insert_count']}")

    query = "Milvus 检索如何结合租户权限和 metadata filter？"
    all_hits = search(client, query, limit=4)
    print_hits("Search without filter", all_hits)

    filtered_hits = search(
        client,
        query,
        limit=4,
        filter_expr="tenant_id == 'team_a' and source == 'handbook'",
    )
    print_hits("Search with tenant/source filter", filtered_hits)

    print(f"\nrecall@3 on tiny eval set: {recall_at_k(client, k=3):.2f}")
    print(f"Milvus Lite database: {DB_PATH}")


if __name__ == "__main__":
    main()
