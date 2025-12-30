import argparse
import os
from pathlib import Path

import ray.data as rd

from llm_data_pipeline.dedup.minhash import VectorizedMinHash


class MinHashCompute:
    def __init__(self):
        # Initialized once per worker process (or actor)
        self.vm = VectorizedMinHash(k=128)

    def __call__(self, batch: dict) -> dict:
        texts = batch.get("text", [])
        out_sigs = []
        out_lens = []

        for text in texts:
            safe_text = str(text) if text is not None else ""
            # Use vectorized computation
            sig = self.vm.compute_signature(safe_text)
            out_sigs.append(sig)
            out_lens.append(len(safe_text))

        batch["minhash_sig"] = out_sigs
        batch["length"] = out_lens
        return batch


def run_minhash(args) -> dict:
    """Computes MinHash signatures for dedup."""
    base_out = getattr(args, "output_dir", "./outputs/dev")
    input_path = Path(getattr(args, "input", f"{base_out}/cleaned_parquet"))
    output_dir = Path(base_out)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    ds = rd.read_parquet(str(input_path.absolute()))

    limit = getattr(args, "limit", 0)
    if limit > 0:
        ds = ds.limit(limit)

    # Use map_batches with Class for stateful initialization (avoid pickling large arrays repeatedly)
    # compute = MinHashCompute() # Ray handles class instantiation if passed as type?
    # Actually for map_batches, we can pass the class directly and compute=ray.data.ActorPoolStrategy?
    # Or just simple map_batches(MinHashCompute, compute=...)
    # Ray data documentation says: map_batches(MyClass, batch_size=..., compute=...)

    # Determine concurrency
    concurrency_arg = getattr(args, "concurrency", None)
    if concurrency_arg:
        concurrency = concurrency_arg
    else:
        # Limit concurrency default to avoid resource exhaustion on local/small setups
        # But allow scaling if ray is clustered (though cpu_count is local)
        # For now, keep the safe default but raise it slightly or leave as auto if None is preferred?
        # User wants "distributed", so let's default to (cpu_count - 1) or similar if not specified?
        # Or just use the old safe default but allow override.
        concurrency = min(4, os.cpu_count() or 1)

    batch_size = getattr(args, "batch_size", 4096)

    ds_sig = ds.map_batches(
        MinHashCompute,
        batch_size=batch_size,
        concurrency=concurrency,
    )

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
