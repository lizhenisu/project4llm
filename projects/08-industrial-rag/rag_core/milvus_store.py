from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from rag_core.config import RagConfig
from rag_core.text_utils import chunk_id, content_hash, now_ms, sparse_embedding
from rag_core.types import Chunk, SearchHit


OUTPUT_FIELDS = [
    "tenant_id",
    "doc_id",
    "doc_version",
    "chunk_index",
    "source_type",
    "source_uri",
    "title",
    "text",
    "language",
    "acl_groups",
    "created_at",
    "updated_at",
    "is_active",
    "embedding_model",
    "embedding_dim",
    "content_hash",
    "metadata",
]


def connect(config: RagConfig) -> MilvusClient:
    kwargs: dict[str, Any] = {}
    if config.milvus_token:
        kwargs["token"] = config.milvus_token
    return MilvusClient(uri=config.milvus_uri, **kwargs)


def create_schema(config: RagConfig):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
    schema.add_field("doc_version", DataType.INT64)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("source_type", DataType.VARCHAR, max_length=32)
    schema.add_field("source_uri", DataType.VARCHAR, max_length=512)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("text", DataType.VARCHAR, max_length=8192)
    schema.add_field("language", DataType.VARCHAR, max_length=16)
    schema.add_field(
        "acl_groups",
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=32,
        max_length=64,
    )
    schema.add_field("created_at", DataType.INT64)
    schema.add_field("updated_at", DataType.INT64)
    schema.add_field("is_active", DataType.BOOL)
    schema.add_field("embedding_model", DataType.VARCHAR, max_length=128)
    schema.add_field("embedding_dim", DataType.INT64)
    schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
    schema.add_field("text_dense_vector", DataType.FLOAT_VECTOR, dim=config.embedding_dim)
    schema.add_field("bm25_sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("image_dense_vector", DataType.FLOAT_VECTOR, dim=config.image_embedding_dim)
    schema.add_field("metadata", DataType.JSON)
    return schema


def create_index_params() -> Any:
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="text_dense_vector",
        index_name="text_dense_hnsw",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 100},
    )
    index_params.add_index(
        field_name="bm25_sparse_vector",
        index_name="bm25_sparse_inverted",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.2},
    )
    index_params.add_index(
        field_name="image_dense_vector",
        index_name="image_dense_hnsw",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 100},
    )
    return index_params


def ensure_collection(client: MilvusClient, config: RagConfig, *, reset: bool = False) -> None:
    if reset and client.has_collection(config.collection_name):
        client.drop_collection(config.collection_name)

    if not client.has_collection(config.collection_name):
        client.create_collection(
            collection_name=config.collection_name,
            schema=create_schema(config),
            index_params=create_index_params(),
        )

    client.load_collection(config.collection_name)


def build_filter_expr(
    *,
    tenant_id: str,
    allowed_acl_groups: list[str] | None = None,
    source_types: list[str] | None = None,
    active_only: bool = True,
    doc_version: int | None = None,
) -> str:
    clauses = [f'tenant_id == "{tenant_id}"']
    if active_only:
        clauses.append("is_active == true")
    if doc_version is not None:
        clauses.append(f"doc_version == {doc_version}")
    if source_types:
        quoted = ", ".join(f'"{item}"' for item in source_types)
        clauses.append(f"source_type in [{quoted}]")
    if allowed_acl_groups:
        quoted = ", ".join(f'"{group}"' for group in allowed_acl_groups)
        clauses.append(f"ARRAY_CONTAINS_ANY(acl_groups, [{quoted}])")
    return " and ".join(clauses)


def chunk_to_entity(
    chunk: Chunk,
    *,
    dense_vector: list[float],
    image_vector: list[float],
    embedding_model: str,
    embedding_dim: int,
) -> dict[str, Any]:
    timestamp = now_ms()
    return {
        "id": chunk_id(chunk),
        "tenant_id": chunk.tenant_id,
        "doc_id": chunk.doc_id,
        "doc_version": chunk.doc_version,
        "chunk_index": chunk.chunk_index,
        "source_type": chunk.source_type,
        "source_uri": chunk.source_uri,
        "title": chunk.title,
        "text": chunk.text,
        "language": chunk.language,
        "acl_groups": chunk.acl_groups,
        "created_at": timestamp,
        "updated_at": timestamp,
        "is_active": True,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "content_hash": content_hash(chunk.text),
        "text_dense_vector": dense_vector,
        "bm25_sparse_vector": sparse_embedding(chunk.text),
        "image_dense_vector": image_vector,
        "metadata": chunk.metadata,
    }


def upsert_entities(
    client: MilvusClient,
    *,
    collection_name: str,
    entities: list[dict[str, Any]],
) -> int:
    if not entities:
        return 0
    result = client.upsert(collection_name=collection_name, data=entities)
    return int(result.get("upsert_count", result.get("insert_count", len(entities))))


def _hit_to_search_hit(hit: dict[str, Any]) -> SearchHit:
    entity = hit["entity"]
    return SearchHit(
        id=str(hit["id"]),
        score=float(hit["distance"]),
        text=entity["text"],
        doc_id=entity["doc_id"],
        title=entity["title"],
        source_uri=entity["source_uri"],
        source_type=entity["source_type"],
        chunk_index=int(entity["chunk_index"]),
        tenant_id=entity["tenant_id"],
        acl_groups=list(entity.get("acl_groups") or []),
        metadata=entity.get("metadata") or {},
    )


def dense_search(
    client: MilvusClient,
    *,
    collection_name: str,
    query_vector: list[float],
    filter_expr: str,
    limit: int,
) -> list[SearchHit]:
    result = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field="text_dense_vector",
        filter=filter_expr,
        limit=limit,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
        output_fields=OUTPUT_FIELDS,
    )
    return [_hit_to_search_hit(hit) for hit in result[0]]


def hybrid_search(
    client: MilvusClient,
    *,
    collection_name: str,
    query_vector: list[float],
    query_sparse: dict[int, float],
    filter_expr: str,
    limit: int,
) -> list[SearchHit]:
    dense_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="text_dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=max(limit, 20),
        expr=filter_expr,
    )
    sparse_req = AnnSearchRequest(
        data=[query_sparse],
        anns_field="bm25_sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}},
        limit=max(limit, 20),
        expr=filter_expr,
    )
    result = client.hybrid_search(
        collection_name=collection_name,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(),
        limit=limit,
        output_fields=OUTPUT_FIELDS,
    )
    return [_hit_to_search_hit(hit) for hit in result[0]]


def image_search(
    client: MilvusClient,
    *,
    collection_name: str,
    image_query_vector: list[float],
    filter_expr: str,
    limit: int,
) -> list[SearchHit]:
    result = client.search(
        collection_name=collection_name,
        data=[image_query_vector],
        anns_field="image_dense_vector",
        filter=filter_expr,
        limit=limit,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
        output_fields=OUTPUT_FIELDS,
    )
    return [_hit_to_search_hit(hit) for hit in result[0]]


def fetch_by_ids(
    client: MilvusClient,
    *,
    collection_name: str,
    ids: Iterable[str],
) -> dict[str, SearchHit]:
    id_list = list(ids)
    if not id_list:
        return {}
    quoted = ", ".join(f'"{item}"' for item in id_list)
    rows = client.query(
        collection_name=collection_name,
        filter=f"id in [{quoted}]",
        output_fields=["id", *OUTPUT_FIELDS],
    )
    return {
        row["id"]: SearchHit(
            id=row["id"],
            score=0.0,
            text=row["text"],
            doc_id=row["doc_id"],
            title=row["title"],
            source_uri=row["source_uri"],
            source_type=row["source_type"],
            chunk_index=int(row["chunk_index"]),
            tenant_id=row["tenant_id"],
            acl_groups=list(row.get("acl_groups") or []),
            metadata=row.get("metadata") or {},
        )
        for row in rows
    }
