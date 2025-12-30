import argparse
from pathlib import Path

import ray
import ray.data as rd

from llm_data_pipeline.dedup.dedup import ray_global_dedup


def parse_args():
    p = argparse.ArgumentParser("llm_data_pipeline.dedup.run_clustering")
    p.add_argument("--input", default="./outputs/dev/minhash_parquet", help="Input path (minhash output)")
    p.add_argument("--output-dir", default="./outputs/dev", help="Output parent directory")
    p.add_argument("--ray-address", default=None)
    p.add_argument("--rows-per-band", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    print(f"Reading from {args.input}...")
    ds = rd.read_parquet(args.input)

    # Rename 'minhash_sig' to 'signature' to match dedup.py expectation
    # and ensure it is a list (pyarrow might read it as numpy array or list)
    def prep_signature(row):
        return {"signature": row["minhash_sig"]}

    ds_for_dedup = ds.map(prep_signature)

    print("Running ray_global_dedup logic...")
    # We need to pass the dataset with 'signature', 'doc_id', 'ts', 'length' (optional)
    # The original ds likely has doc_id.
    # We combine original columns with prepared signature
    ds_combined = ds.map(lambda r: {**r, "signature": r["minhash_sig"]})

    result = ray_global_dedup(ds_combined, rows_per_band=args.rows_per_band)

    keep_set = result["keep_set"]
    print(f"Dedup finished. Kept {result['kept_docs']} / {result['total_docs']} docs.")

    # Filter original dataset
    # Note: keep_set is a set of doc_ids. For large datasets, broadcasting a large set might be heavy.
    # But for this task it's sufficient.

    # Converting keep_set to a broadcastable object or handling it efficiently
    # For very large sets, join strategies or Bloom filters are better.
    # Here proceed with simple filter if set fits in memory.

    # Collect keep_set to driver (assuming it fits in memory for this scale)
    # For large scale, you'd broadcast this or use a distributed map-side join.
    keep_set_broad = ray.put(keep_set)

    def is_kept(row):
        return row["doc_id"] in ray.get(keep_set_broad)

    # Use map_batches to filter efficiently if we were pushing this down,
    # but for simple filter, we can use filter().
    # However, using a lambda with ray.get inside might be slow per row.
    # A better way for Ray is to carry the set in the actor or mapper init.
    # But for simplicity here:

    # We will use map_batches to reduce ray.get overhead
    ds_final = ds.map_batches(
        lambda batch: {k: [v[i] for i in range(len(v)) if batch["doc_id"][i] in keep_set] for k, v in batch.items()},
        batch_format="dict",
    )

    out_dir = Path(args.output_dir) / "deduped_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing deduped data to {out_dir}...")
    ds_final.write_parquet(str(out_dir))
    print("Done.")


if __name__ == "__main__":
    main()
