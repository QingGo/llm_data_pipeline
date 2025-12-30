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


def run_clustering(args) -> dict:
    """Clustering & Dedup Step"""
    base_out = getattr(args, "output_dir", "./outputs/dev")
    input_path = Path(getattr(args, "input", f"{base_out}/minhash_parquet"))
    output_dir = Path(base_out)
    rows_per_band = getattr(args, "rows_per_band", 4)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist for clustering.")

    print(f"Reading from {input_path}...")
    ds = rd.read_parquet(str(input_path.absolute()))

    limit = getattr(args, "limit", 0)
    if limit > 0:
        print(f"DEBUG: Limiting clustering input to {limit} records.")
        ds = ds.limit(limit)

    # Rename 'minhash_sig' to 'signature' if needed
    # The previous code had a map just for this.
    # Our minhash_step produces "minhash_sig".
    # ray_global_dedup expects "signature" in row dict?
    # Let's check dedup.py: make_band_rows uses row["signature"].
    # So we must ensure it exists.

    # We can do a rename-only map or just rely on the fact that minhash_step added minhash_sig
    # and we can map it to signature.

    ds = ds.map(lambda r: {**r, "signature": r.get("minhash_sig", r.get("signature"))})

    print("Running ray_global_dedup logic...")
    result = ray_global_dedup(ds, rows_per_band=rows_per_band)

    keep_set = result["keep_set"]
    print(f"Dedup finished. Kept {result['kept_docs']} / {result['total_docs']} docs.")

    # Filter original dataset
    # We use map_batches to filter efficiently
    # Note: we need to filter AND write to deduped_parquet.
    # The output of dedup/run.py writes to deduped_parquet.

    keep_set_broad = ray.put(keep_set)

    ds_final = ds.map_batches(
        lambda batch: {k: [v[i] for i in range(len(v)) if batch["doc_id"][i] in keep_set] for k, v in batch.items()},
        batch_format="dict",
    )
    # Remove signature/minhash_sig to save space? Optional.
    # Usually we want clean text for next steps.

    out_dir = output_dir / "deduped_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing deduped data to {out_dir}...")
    ds_final.write_parquet(str(out_dir))

    final_count = ds_final.count()
    print("Done.")

    return {
        "input_docs": result["total_docs"],
        "kept_docs": result["kept_docs"],
        "removed_docs": result["removed_docs"],
        "dedup_rate": result["dedup_rate"],
        "output_path": str(out_dir),
    }


def main():
    args = parse_args()
    ray.init(address=args.ray_address, ignore_reinit_error=True)
    run_clustering(args)
