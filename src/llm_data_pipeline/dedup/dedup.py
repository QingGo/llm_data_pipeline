import hashlib
import itertools
from pathlib import Path
from typing import Any

import ray
import ray.data as rd

from llm_data_pipeline.dedup.minhash import char_ngrams, datasketch_minhash


# ---------- LSH banding (Map) ----------
def band_hash(vals: list[int]) -> str:
    # 稳定 hash：把 band 的 ints 打包成 bytes 再 sha1
    b = (",".join(map(str, vals))).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


def make_band_rows(row: dict[str, Any], rows_per_band: int) -> list[dict[str, Any]]:
    sig = row["signature"]
    num_perm = len(sig)
    assert num_perm % rows_per_band == 0
    out = []
    length = row.get("length", len(row.get("text", "")))
    ts = row.get("ts", 0)
    for band_id in range(num_perm // rows_per_band):
        s = band_id * rows_per_band
        e = s + rows_per_band
        out.append(
            {
                "band_id": band_id,
                "band_hash": band_hash(sig[s:e]),
                "doc_id": row["doc_id"],
                "ts": ts,
                "length": length,
            }
        )
    return out


# ---------- bucket -> edges (Reduce) ----------
def bucket_to_pairs(batch) -> dict[str, list[dict[str, Any]]]:
    """
    batch: 一个桶内的记录（同 band_id, band_hash）
    输出：该桶内所有 doc_id 两两组合的边，包装在字典中
    """
    # batch 是 pyarrow table / pandas df 都行，这里用最通用的转 dict
    docs = []
    for r in batch.to_pylist():
        docs.append((r["doc_id"], r["ts"], r["length"]))

    if len(docs) < 2:
        return {"edges": []}

    # 可选：桶太大直接降级，避免 O(n^2) 炸裂
    # docs = docs[:500]

    edges = []
    for (a, _, _), (b, _, _) in itertools.combinations(docs, 2):
        u, v = (a, b) if a < b else (b, a)
        edges.append({"u": u, "v": v})
    return {"edges": edges}


# ---------- union-find connected components ----------
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def pick_canonical(members: list[tuple[str, int, int]]) -> str:
    # members: [(doc_id, ts, length), ...]
    members.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
    return members[0][0]


def ray_global_dedup(
    docs_ds: rd.Dataset,
    rows_per_band: int = 4,
):
    # Map: 展开 band 行
    band_rows = docs_ds.flat_map(lambda r: make_band_rows(r, rows_per_band))

    # Reduce: groupBy 桶 -> 产出候选边
    # bucket_to_pairs returns {"edges": [...]}, we need to unwrap it
    edges_ds = (
        band_rows.groupby(["band_id", "band_hash"])
        .map_groups(bucket_to_pairs, batch_format="pyarrow")
        .flat_map(lambda row: row["edges"])
    )

    # 边去重（同一对可能在多个桶命中）
    edges_ds = edges_ds.groupby(["u", "v"]).count().drop_columns(["count()"])

    # 统计信息
    total_docs = docs_ds.count()
    total_edges = edges_ds.count()

    # --- CC（本地版本：把 edges 拉回 driver）---
    edges = edges_ds.take_all()  # ⚠️ 边很大时不要这么做
    uf = UnionFind()
    for e in edges:
        uf.union(e["u"], e["v"])

    # 每个 doc -> root
    docs_meta = docs_ds.select_columns(["doc_id", "ts", "length"]).take_all()
    comp: dict[str, list[tuple[str, int, int]]] = {}
    for r in docs_meta:
        doc_id = r["doc_id"]
        ts = r.get("ts", 0)
        length = r.get("length", 0)
        root = uf.find(doc_id)
        comp.setdefault(root, []).append((doc_id, ts, length))

    # 每个分量选 canonical，生成删除集合
    keep = set()
    for _, members in comp.items():
        keep.add(pick_canonical(members))

    kept_docs = len(keep)
    removed_docs = total_docs - kept_docs
    dedup_rate = removed_docs / total_docs if total_docs else 0.0

    return {
        "duplicate_pairs_sample": edges[:50],  # 示例
        "total_docs": total_docs,
        "total_candidate_pairs": total_edges,
        "kept_docs": kept_docs,
        "removed_docs": removed_docs,
        "dedup_rate": dedup_rate,
        "keep_set": keep,
    }


if __name__ == "__main__":
    from llm_data_pipeline.core import PipelineLogger, setup_logging

    # 配置本地日志
    logger = setup_logging(Path("./outputs/test"), "dedup")
    logger = PipelineLogger.get()

    # 1. 准备测试数据
    docs = [
        {"doc_id": "doc1", "text": "今天天气不错，适合出去玩", "ts": 100},
        {"doc_id": "doc2", "text": "今天天气真好，适合去郊游", "ts": 101},  # 近似
        {"doc_id": "doc3", "text": "完全不同的一句话", "ts": 102},
        {"doc_id": "doc4", "text": "今天天气不错，适合出去玩", "ts": 103},  # 完全重复
    ]

    # 2. 计算 MinHash 签名
    # 注意：实际生产中这一步通常是一个 map_batches 算子
    # 这里为了演示 ray_global_dedup，手动先算好 signature
    processed_docs = []
    for d in docs:
        # 生成 shingles
        shingles = char_ngrams(d["text"], n=5)
        # 生成 MinHash 对象
        m = datasketch_minhash(shingles, k=128)
        # 提取签名 (hashvalues)
        d["signature"] = m.hashvalues.tolist()
        d["length"] = len(d["text"])
        processed_docs.append(d)

    # 3. 启动 Ray 并创建 Dataset
    ray.init()
    ds = ray.data.from_items(processed_docs)

    # 4. 执行去重
    # rows_per_band 越小，band 数越多，召回率越高（越容易判定为相似），但也越慢
    result = ray_global_dedup(ds, rows_per_band=4)

    # 5. 打印结果
    logger.info("=== Dedup Result ===")
    logger.info(f"Total docs: {result['total_docs']}")
    logger.info(f"Kept docs: {result['kept_docs']}")
    logger.info(f"Removed docs: {result['removed_docs']}")
    logger.info(f"Dedup rate: {result['dedup_rate']:.2%}")
    logger.info(f"Duplicate pairs sample: {result['duplicate_pairs_sample']}")
    logger.info(f"Keep set size: {len(result['keep_set'])}")
