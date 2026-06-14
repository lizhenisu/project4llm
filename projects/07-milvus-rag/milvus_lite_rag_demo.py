from __future__ import annotations

# =============================================================================
# milvus_lite_rag_demo.py
# 教学用 RAG 最小闭环：schema → insert → search → filter → recall@k 评估
# =============================================================================
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
# 向量维度。这里用 64 维是教学简化；生产中 BGE-M3 用 1024 维，OpenAI 用 1536/3072 维。
# 维度越高表达能力越强，但存储和计算成本也越大。

COLLECTION_NAME = "rag_chunks_demo"
DB_PATH = Path(__file__).with_name("milvus_lite_demo.db")
TENANT_ID_MAX_LENGTH = 32
SOURCE_MAX_LENGTH = 32
DOC_ID_MAX_LENGTH = 64
TEXT_MAX_LENGTH = 512
# 以上 max_length 是在 Milvus schema 中声明的 VARCHAR 字段最大长度。
# VARCHAR 字段必须声明 max_length，Milvus 用这个值优化存储和索引。
# 生产中 text 字段通常设置为 8192 或更大，这里用 512 纯粹是教学数据短。

# =============================================================================
# Milvus schema 设计：为什么必须显式定义？（面试高频考点）
# =============================================================================
# 见下方 reset_collection() 函数的详细注释。


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


# ---------------------------------------------------------------------------
# 1. 分词：把文本切成一串 token
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """将文本切分为 token 列表。

    规则很简单：
    - 英文字母/数字 和 中文字符（Unicode 范围 \\u4e00-\\u9fff）视为 token 的组成部分
    - 遇到空格、标点、换行等分隔符时，把当前积累的 token 切割出来
    - 全部小写化，避免 "Milvus" 和 "milvus" 产生不同的 hash

    这不是真正的 NLP tokenizer（BPE/WordPiece/SentencePiece），
    而是教学用的最小化分词，让 hash_embedding 能按语义单元（而非单个字符）hash。

    示例：
      "RAG 系统"                      → ["rag", "系统"]
      "Milvus collection 保存向量"    → ["milvus", "collection", "保存向量"]
    """
    normalized = text.lower()
    tokens: list[str] = []
    current: list[str] = []
    for char in normalized:
        # isalnum(): 英文字母 a-z、数字 0-9
        # \\u4e00-\\u9fff: 中文字符 CJK Unified Ideographs 基本区
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            current.append(char)
        elif current:
            # 遇到分隔符，将当前积累的连续字母/数字/汉字作为一个 token
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


# ---------------------------------------------------------------------------
# 2. 确定性 hash embedding（教学简化，不下载模型）
# ---------------------------------------------------------------------------
def hash_embedding(text: str, dim: int = DIM) -> list[float]:
    """用确定性 hash 生成伪向量 embedding（教学用，不下载模型）。

    这个函数的目的是让教学 demo 不需要下载大模型就能跑通整条 RAG 检索链路。
    生产的 RAG 不能这么干：必须用 BGE-M3、text-embedding-3 等真实模型。

    算法步骤（每个 token 在向量空间中"投票"）：

    1. blake2b 哈希 → 把 token 变成一个确定性的 8 字节指纹
    2. 前 4 字节决定"落在哪个桶"（bucket = hash % dim）
    3. 第 5 字节的奇偶决定"投票方向"（+1 或 -1）
    4. 所有 token 投票完后，把向量归一化到单位长度

    为什么归一化？见函数底部注释。
    """
    vector = [0.0] * dim
    for token in tokenize(text):
        # blake2b：比 MD5/SHA1 更快的密码学哈希，digest_size=8 输出 8 字节
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()

        # ------------------------------------------------------------------
        # int.from_bytes(digest[:4], "little") 解释：
        #
        # digest 是 bytes 对象（byte string），digest[:4] 取前 4 个字节。
        # from_bytes 把 4 个字节按指定字节序解析为整数。
        #
        # "little" = 小端序（little-endian）：低地址存整数低位。
        #
        #   举例：4 字节序列 [0x01, 0x00, 0x00, 0x00]
        #     小端解析 → 整数 1×2⁰ + 0×2⁸ + 0×2¹⁶ + 0×2²⁴ = 1
        #     大端解析 → 整数 0×2⁰ + 0×2⁸ + 0×2¹⁶ + 1×2²⁴ = 16777216
        #
        #   选 little 或 big 不影响算法正确性，只要一致即可。
        #   大多数 CPU（x86、ARM）天然使用小端序。
        # ------------------------------------------------------------------
        bucket = int.from_bytes(digest[:4], "little") % dim

        # digest[4] 是第 5 字节（索引从 0 开始）
        # % 2 判断奇偶 → 奇数为 -1.0，偶数为 +1.0
        # 这样不同 token 的投票可能互相抵消或叠加，形成有区分度的向量
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    # ------------------------------------------------------------------
    # 向量归一化（normalization）
    #
    # L2 norm（欧几里得范数）：各分量的平方和取根号 = 向量的几何长度
    #   例如向量 [3, 4] 的 norm = sqrt(3²+4²) = 5
    #
    # 归一化：每个分量除以 norm，使向量变成单位向量（长度=1）。
    #
    # 为什么要归一化？
    #
    # 1. COSINE metric 下的等效计算：
    #    cos(A,B) = A·B / (|A| × |B|)
    #    如果 |A| = |B| = 1，则 cos(A,B) = A·B（点积），计算更简单。
    #
    # 2. 消除文本长度偏差：
    #    长文本的 hash embedding 天然有更多 token 投票，向量长度会更大。
    #    归一化后只保留方向信息，不会因为文本长就"更相似"。
    #
    # 3. 数学直觉（欧几里得距离）：
    #    向量 B = (0, 10)，向量 A = (1, 0)，和原点 C = (0, 0) 比较：
    #      B 到 C 的距离² = 0² + 10² = 100  → 距离 = 10
    #      A 到 C 的距离² = 1² + 0²  = 1    → 距离 = 1
    #    所以 A 离 C 更近。这是欧几里得距离，考虑了绝对位置。
    #    归一化后，A 变成 (1,0)，B 变成 (0,1)，两者与原点内积都等于 0
    #    （cosine=0，向量正交），不再有长度带来的偏差。
    # ------------------------------------------------------------------
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        # 空文本或全零向量，无法归一化，直接返回零向量
        return vector
    return [value / norm for value in vector]


# ---------------------------------------------------------------------------
# 3. 构建 Milvus entity（文档 + vector）
# ---------------------------------------------------------------------------
def build_entities() -> list[dict]:
    """构建 Milvus entity：把文档的 source/doc_id/tenant_id/text 拼起来做 hash embedding。"""
    return [
        {
            **doc,
            "vector": hash_embedding(
                f"{doc['source']} {doc['doc_id']} {doc['tenant_id']} {doc['text']}"
            ),
        }
        for doc in DOCUMENTS
    ]


# ---------------------------------------------------------------------------
# 4. Schema 设计：面试必问的 collection 定义
# ---------------------------------------------------------------------------
def reset_collection(client: MilvusClient) -> None:
    """创建（或重建）rag_chunks_demo collection。

    =======================================================================
    面试必问：Milvus 里 schema 需要预先定义吗？能像 MongoDB 那样随手 insert 吗？
    =======================================================================

    答案：必须预先定义 schema。Milvus 是强 schema 向量数据库，和 MongoDB
    的 dynamic schema 完全不同。

    面试怎么回答：
    "Milvus 需要显式定义 schema。因为向量字段的维度（dim）在建 collection
     时就确定了，FLOAT_VECTOR 的索引也需要提前知道维度；VARCHAR 字段
     必须声明 max_length 供引擎做内存预分配；索引用 HNSW 还是 IVF 也是在
     建索引时绑定的。这和 MongoDB 不一样——MongoDB 的动态 schema 允许随时
     插入新字段，但向量检索场景更需要可预测的性能和存储布局。"

    具体原因：
    1. 向量字段的维度必须在建 collection 时确定。FLOAT_VECTOR 没有固定 dim，
       索引构建和 ANN 搜索都需要提前知道向量的维度和 metric type。
    2. VARCHAR 字段必须声明 max_length，Milvus 用这个做内存预分配。
    3. 索引参数（HNSW 的 M/efConstruction、IVF 的 nlist）在建索引时绑定。
    4. enable_dynamic_field=False 是生产默认：不让脏字段悄悄进入库，
       也可以防止字段拼写错误被当成 dynamic field 吞掉。

    如果只是快速实验，可以用 enable_dynamic_field=True，但生产不建议。

    本 demo 的字段设计：
    - id (INT64, primary key)：主键，唯一标识一条 chunk
    - vector (FLOAT_VECTOR, dim=64)：chunk 的向量表示
    - tenant_id (VARCHAR, max_length=32)：租户隔离
      生产中查询时必须带 tenant filter，且 tenant_id 应来自认证服务，
      不能信任用户端传入的值。
    - source (VARCHAR, max_length=32)：文档来源类型
    - doc_id (VARCHAR, max_length=64)：所属文档 ID
      同一篇文档的多个 chunk 共享同一个 doc_id。
    - text (VARCHAR, max_length=512)：chunk 原文
    """
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # auto_id=False：我们自己控制主键值（用文档数据的 id 字段）
    # enable_dynamic_field=False：不接收 schema 里没定义的字段
    #   如果设为 True，insert 时传了 schema 没有的字段也不会报错，
    #   Milvus 会自动存到内部 JSON 字段。生产慎用。
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    # 主键字段：每条 chunk 的唯一标识
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )

    # 向量字段：chunk 的 embedding
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=DIM,
    )

    # 以下是 metadata 字段 —— 面试重点：向量检索 + metadata filter 缺一不可
    schema.add_field(
        field_name="tenant_id",
        datatype=DataType.VARCHAR,
        max_length=TENANT_ID_MAX_LENGTH,
        # is_partition_key=True,  # 生产中可以设为分区键加速过滤，教学 demo 先不设
    )
    schema.add_field(
        field_name="source",
        datatype=DataType.VARCHAR,
        max_length=SOURCE_MAX_LENGTH,
    )
    # doc_id：同一篇文档的所有 chunk 共享
    schema.add_field(
        field_name="doc_id",
        datatype=DataType.VARCHAR,
        max_length=DOC_ID_MAX_LENGTH,
    )
    # text：chunk 正文。检索命中后可以直接展示或进入 LLM prompt
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=TEXT_MAX_LENGTH,
    )

    # metric_type="COSINE"：余弦相似度
    # Milvus 支持三种：COSINE / IP（内积）/ L2（欧几里得距离）
    # COSINE 只关心方向不关心长度，适合文本语义比较
    # 如果向量已经归一化（本 demo 的 hash_embedding 已经做了），COSINE 等价于 IP
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        metric_type="COSINE",
    )

    # 在生产中，created collection 之后通常还会建索引：
    #
    # index_params = MilvusClient.prepare_index_params()
    # index_params.add_index(
    #     field_name="vector",
    #     index_type="HNSW",
    #     metric_type="COSINE",
    #     params={"M": 16, "efConstruction": 100},
    # )
    # client.create_index(
    #     collection_name=COLLECTION_NAME,
    #     index_params=index_params,
    # )
    #
    # Milvus Lite 在小数据量下可以省略显式建索引，会用 FLAT（暴力搜索）。
    # 但生产必须建索引：HNSW 在百万级数据下可以把延迟从秒级降到毫秒级。


# ---------------------------------------------------------------------------
# 5. 检索辅助函数
# ---------------------------------------------------------------------------
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
    # filter_expr 是 Milvus 的标量过滤表达式，语法类似 SQL WHERE
) -> list[dict]:
    result = client.search(
        collection_name=COLLECTION_NAME,
        data=[hash_embedding(query)],
        limit=limit,
        filter=filter_expr,
        output_fields=["text", "doc_id", "tenant_id", "source"],
    )
    return result[0]


# ---------------------------------------------------------------------------
# 6. 评估：recall@k
# ---------------------------------------------------------------------------
def recall_at_k(client: MilvusClient, k: int = 3) -> float:
    """计算 recall@k：对每个 query，检查 topK 候选中是否包含任意一个正确答案。

    recall@k = 命中的 query 数 / 总 query 数

    面试常问：recall@k 和 MRR 的区别？

    recall@k：只看"有没有命中"，不关心排名。k=3 时第 1 名和第 3 名命中都算命中。

    MRR (Mean Reciprocal Rank)：看"命中排第几"，排名越靠前分数越高。
      命中在第 1 名 → 1/1 = 1.0
      命中在第 3 名 → 1/3 ≈ 0.33
      没命中       → 0

    举例（IR 101）：
      query1 正确答案排第 1  → recall@3=hit, RR=1.0
      query2 正确答案排第 3  → recall@3=hit, RR≈0.33
      query3 正确答案没进 top3 → recall@3=miss, RR=0
      总 recall@3 = 2/3 ≈ 0.67
      总 MRR = (1.0 + 0.33 + 0) / 3 ≈ 0.44

    生产中除了 recall@k 和 MRR，还需要关注：
    - nDCG@k：支持多级相关性标注（非常相关 > 部分相关 > 不相关）
    - 权限泄露率：无权限租户的 chunk 是否意外出现在结果中
    - p95/p99 延迟：embedding + search + rerank 各段耗时分布
    """
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


# ---------------------------------------------------------------------------
# 7. 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    # MilvusClient 使用本地文件路径作为 URI，启动一个嵌入式的 Milvus Lite 实例。
    # 不需要 Docker，不需要独立服务进程，非常适合教学和本地开发。
    client = MilvusClient(str(DB_PATH))
    reset_collection(client)

    # 插入文档：把模拟的 6 条知识库 chunk 写入 Milvus
    entities = build_entities()
    insert_result = client.insert(collection_name=COLLECTION_NAME, data=entities)
    print(f"Inserted rows: {insert_result['insert_count']}")

    # ==================================================
    # 检索演示 1：不加 filter 的普通向量检索
    # ==================================================
    query = "Milvus 检索如何结合租户权限和 metadata filter？"
    all_hits = search(client, query, limit=4)
    print_hits("Search without filter", all_hits)

    # ==================================================
    # 检索演示 2：带 tenant_id 和 source 的 filtered search
    # ==================================================
    # filter 表达式：tenant_id == 'team_a' and source == 'handbook'
    # 加上 filter 后 team_b 的数据直接被排除在候选集之外。
    filtered_hits = search(
        client,
        query,
        limit=4,
        filter_expr="tenant_id == 'team_a' and source == 'handbook'",
    )
    # 对比：finance-001 (team_b) 被排除，milvus-ops-001 (runbook) 被 source 过滤
    print_hits("Search with tenant/source filter", filtered_hits)

    # ==================================================
    # 评估演示：recall@3
    # ==================================================
    print(f"\nrecall@3 on tiny eval set: {recall_at_k(client, k=3):.2f}")
    print(f"Milvus Lite database: {DB_PATH}")


if __name__ == "__main__":
    main()
