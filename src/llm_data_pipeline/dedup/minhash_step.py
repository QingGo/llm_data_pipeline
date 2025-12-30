import argparse
import logging
import os
from pathlib import Path

import ray.data as rd

from llm_data_pipeline.core import (
    PipelineConfig,
    resolve_io_paths,
    run_step_entrypoint,
)
from llm_data_pipeline.dedup.minhash import VectorizedMinHash

logger = logging.getLogger(__name__)


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

        # [REFACTOR] Standardize column name to 'signature'
        batch["signature"] = out_sigs
        batch["length"] = out_lens
        return batch


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", default=None, help="Input directory")


def run_minhash(config: PipelineConfig, **kwargs) -> dict:
    """Computes MinHash signatures for dedup."""
    # Resolve paths
    manual_input = kwargs.get("input")
    if manual_input:
        input_path = Path(manual_input)
        _, output_dir = resolve_io_paths(config, "minhash")
    else:
        input_path, output_dir = resolve_io_paths(config, "minhash", "clean")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    ds = rd.read_parquet(str(input_path.absolute()))

    limit = config.limit
    if limit > 0:
        ds = ds.limit(limit)

    # Determine concurrency
    # We respect config.concurrency if set, else auto.
    concurrency = config.concurrency

    # Logic in previous version was: if None, default to min(4, cpu_count).
    # We can preserve this heuristic if concurrency is None.
    if concurrency is None:
        concurrency = min(4, os.cpu_count() or 1)

    batch_size = config.batch_size

    ds_sig = ds.map_batches(
        MinHashCompute,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        compute=rd.ActorPoolStrategy(size=concurrency),  # pyright: ignore[reportArgumentType]
    )

    out_path = output_dir / "minhash_parquet"
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing minhash results to {out_path}...")
    ds_sig.write_parquet(str(out_path.absolute()))

    count = ds_sig.count()
    logger.info(f"Computed minhash for {count} docs.")

    return {"docs_processed": count, "output_path": str(out_path)}


def main():
    run_step_entrypoint(
        description="MinHash Step",
        run_func=run_minhash,
        add_args_func=add_args,
        step_name="MinHash",
    )
