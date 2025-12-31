import argparse
import os

import ray.data as rd
from ray.data import ActorPoolStrategy

from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    resolve_io_paths,
    run_step_entrypoint,
    validate_input_path,
    write_parquet,
)
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

        # [REFACTOR] Standardize column name to 'signature'
        batch["signature"] = out_sigs
        batch["length"] = out_lens
        return batch


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", default=None, help="Input directory")


def run_minhash(config: PipelineConfig, **kwargs) -> dict:
    """Computes MinHash signatures for dedup."""
    logger = PipelineLogger.get()

    # Resolve paths
    input_path, output_dir = resolve_io_paths(config, "minhash", "clean")

    # Validate input path
    validate_input_path(input_path, "minhash")

    # Log input path and document count
    logger.info(f"Reading MinHash input from: {input_path}")
    ds = rd.read_parquet(str(input_path.absolute()))
    logger.info(f"Found {ds.count()} docs in input path before applying limit")

    limit = config.limit
    if limit > 0:
        logger.info(f"Applying limit of {limit} docs to MinHash input")
        ds = ds.limit(limit)
        logger.info(f"Input docs after limit: {ds.count()}")

    # Determine concurrency
    # We respect config.concurrency if set, else auto.
    concurrency = config.concurrency

    # Logic in previous version was: if None, default to min(4, cpu_count).
    # We can preserve this heuristic if concurrency is None.
    if concurrency is None:
        concurrency = min(4, os.cpu_count() or 1)

    batch_size = config.batch_size

    ds_sig = ds.map_batches(
        MinHashCompute,
        batch_size=batch_size,
        compute=ActorPoolStrategy(size=concurrency),
    )

    out_path = output_dir / "minhash_parquet"
    write_parquet(ds_sig, out_path, logger)

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
