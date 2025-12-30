import argparse
from pathlib import Path

import ray.data as rd

from llm_data_pipeline.dedup.minhash import char_ngrams, datasketch_minhash


def compute_minhash(row):
    text = row.get("text", "")
    shingles = char_ngrams(text, n=5)
    m = datasketch_minhash(shingles, k=128)
    # datasketch.MinHash hashvalues is numpy array
    sig = m.hashvalues.tolist()
    # We might want to keep other columns or just doc_id and sig
    return {**row, "minhash_sig": sig, "length": len(text)}


def run_minhash(args) -> dict:
    """Computes MinHash signatures for dedup."""
    base_out = getattr(args, "output_dir", "./outputs/dev")
    input_path = Path(getattr(args, "input", f"{base_out}/cleaned_parquet"))
    output_dir = Path(base_out)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    ds = rd.read_parquet(str(input_path.absolute()))

    # We map to add signature
    # Since minhash computation is CPU bound per doc, simple map is fine.
    # For optimization, map_batches with pandas/numpy vectorization or calling C++ would be faster,
    # but datasketch is python based (with some C).
    # Let's use map for simplicity and consistency with current codebase.

    ds_sig = ds.map(compute_minhash)

    out_path = output_dir / "minhash_parquet"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Writing minhash results to {out_path}...")
    ds_sig.write_parquet(str(out_path.absolute()))

    count = ds_sig.count()
    print(f"Computed minhash for {count} docs.")

    return {"docs_processed": count, "output_path": str(out_path)}


def main():
    p = argparse.ArgumentParser("MinHash Step")
    p.add_argument("--input", default="./outputs/dev/cleaned_parquet")
    p.add_argument("--output-dir", default="./outputs/dev")
    args = p.parse_args()

    import ray

    ray.init()
    run_minhash(args)


if __name__ == "__main__":
    main()
