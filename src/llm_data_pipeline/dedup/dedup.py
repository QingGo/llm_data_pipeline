"""
Global Deduplication Utilities.

This module contains the core logic for Locality Sensitive Hashing (LSH) and graph-based deduplication.
It includes functions for band hashing, generating candidate pairs, and finding connected components
to identify and remove duplicate documents.
"""

import hashlib
import itertools
from typing import Any

import ray
import ray.data as rd


# ---------- LSH banding (Map) ----------
def band_hash(vals: list[int]) -> str:
    """
    Computes a stable hash for a given band (list of integers).
    Uses SHA1 on the string representation of values.
    """
    b = (",".join(map(str, vals))).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


def make_band_rows(row: dict[str, Any], rows_per_band: int) -> list[dict[str, Any]]:
    """
    Generates band rows for LSH.

    Splits the MinHash signature into multiple bands. Each band is hashed to create a bucket key.
    Documents falling into the same bucket (same band hash) are candidate duplicates.

    Args:
        row: Input row containing 'signature', 'doc_id', etc.
        rows_per_band: Number of signature components per band.

    Returns:
        List of dictionary objects representing the bands for this document.
    """
    if "signature" not in row:
        raise ValueError(
            f"Row is missing required 'signature' field. "
            f"Available fields: {list(row.keys())}. "
            f"Please ensure the 'minhash' step was run before 'clustering' step."
        )

    sig = row["signature"]
    num_perm = len(sig)
    assert num_perm % rows_per_band == 0, (
        f"Number of permutations {num_perm} must be divisible by rows_per_band {rows_per_band}"
    )

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
def bucket_to_pairs(batch) -> list[dict[str, Any]]:
    """
    Generates candidate duplicate pairs from a bucket.

    Args:
        batch: A batch of records in the same bucket (same band_id and band_hash).

    Returns:
        A list of edges (pairs of doc_ids) representing potential duplicates.
    """
    # batch 是 pyarrow table / pandas df 都行，这里用最通用的转 dict
    docs = []
    for r in batch.to_pylist():
        docs.append((r["doc_id"], r["ts"], r["length"]))

    if len(docs) < 2:
        return []

    # 可选：桶太大直接降级，避免 O(n^2) 炸裂
    # docs = docs[:500]

    edges = []
    for (a, _, _), (b, _, _) in itertools.combinations(docs, 2):
        u, v = (a, b) if a < b else (b, a)
        edges.append({"u": u, "v": v})
    return edges


# ---------- union-find connected components ----------
class UnionFind:
    """Disjoint Set Union (DSU) data structure for connected components."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        """Finds the representative (root) of the set containing x with path compression."""
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        """Unions the sets containing a and b."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def pick_canonical(members: list[tuple[str, int, int]]) -> str:
    """
    Selects the canonical document from a cluster of duplicates.
    Preference: Higher timestamp -> Larger length -> Lexicographical doc_id.
    """
    # members: [(doc_id, ts, length), ...]
    members.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
    return members[0][0]


def ray_global_dedup(
    docs_ds: rd.Dataset,
    rows_per_band: int = 4,
):
    """
    Performs global deduplication using LSH and Connected Components on Ray Data.

    1. Expands each document into multiple band rows (Map phase).
    2. Groups by band hash to find buckets (Shuffle phase).
    3. Generates candidate pairs (edges) within buckets (Reduce phase).
    4. Computes connected components on the graph of candidate pairs.
    5. Selects one canonical document per component to keep.

    Args:
        docs_ds: Input dataset with signatures.
        rows_per_band: LSH parameter controlling sensitivity.

    Returns:
        Dict containing stats and the set of 'keep' doc_ids.
    """
    # Map: Expand band rows
    band_rows = docs_ds.flat_map(lambda r: make_band_rows(r, rows_per_band))

    # For Ray Data >= 2.5, we need a different approach
    # First, materialize the band rows to get all the data
    all_band_rows = band_rows.take_all()

    # Create a dictionary to group band rows by (band_id, band_hash)
    buckets = {}
    for row in all_band_rows:
        key = (row["band_id"], row["band_hash"])
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(row)

    # Generate edges from each bucket
    all_edges = []
    for bucket_rows in buckets.values():
        # Create a simple batch-like object with to_pylist method
        class SimpleBatch:
            def __init__(self, rows):
                self.rows = rows

            def to_pylist(self):
                return self.rows

        batch = SimpleBatch(bucket_rows)
        edges = bucket_to_pairs(batch)
        all_edges.extend(edges)

    # Create a dataset from the edges
    edges_ds = ray.data.from_items(all_edges)

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
    # Check available columns and select only existing ones
    available_cols = docs_ds.columns() or []
    select_cols = ["doc_id", "length"]
    if "ts" in available_cols:
        select_cols.append("ts")

    docs_meta = docs_ds.select_columns(select_cols).take_all()
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
