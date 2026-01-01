"""
MinHash Step Entry Point.

This module defines the Ray Data processing step for computing MinHash signatures for documents.
It wraps the `VectorizedMinHash` computation and maps it over the dataset.
"""

import os

import ray.data as rd
from ray.data import ActorPoolStrategy

from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    run_step_entrypoint,
    step_wrapper,
)
from llm_data_pipeline.dedup.minhash import VectorizedMinHash


class MinHashCompute:
    """
    Callable class for MinHash computation in Ray Data.
    Initializes the VectorizedMinHash object once per actor/worker.
    """

    def __init__(self):
        # Initialized once per worker process (or actor)
        self.vm = VectorizedMinHash(k=128)

    def __call__(self, batch: dict) -> dict:
        """
        Computes signatures for a batch of texts.
        """
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


def add_args(p) -> None:
    pass


def _process_minhash(ds: rd.Dataset, config: PipelineConfig, **kwargs) -> rd.Dataset:
    """Core MinHash processing function"""
    logger = PipelineLogger.get()

    # Log document count before limit
    logger.info(f"Found {ds.count()} docs in input before applying limit")

    # Determine concurrency
    # We respect config.concurrency if set, else auto.
    concurrency = config.concurrency

    # Logic in previous version was: if None, default to min(4, cpu_count).
    # We can preserve this heuristic if concurrency is None.
    if concurrency is None:
        concurrency = min(4, os.cpu_count() or 1)

    batch_size = config.batch_size

    ds_sig = ds.map_batches(
        MinHashCompute,  # type: ignore
        batch_size=batch_size,
        compute=ActorPoolStrategy(size=concurrency),
    )

    return ds_sig


def run_minhash(config: PipelineConfig, **kwargs) -> dict:
    """Computes MinHash signatures for dedup."""
    stats = step_wrapper(
        step_name="minhash", process_func=_process_minhash, config=config, input_step_name="pii", **kwargs
    )

    # Add minhash-specific stats
    stats.update({"docs_processed": stats["output_count"]})

    return stats


def main():
    run_step_entrypoint(
        description="MinHash Step",
        run_func=run_minhash,
        add_args_func=add_args,
        step_name="MinHash",
    )
