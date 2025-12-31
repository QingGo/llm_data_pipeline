import argparse
from pathlib import Path

import ray.data as rd

from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    resolve_io_paths,
    run_step_entrypoint,
    validate_input_path,
)
from llm_data_pipeline.dedup.dedup import ray_global_dedup


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", default=None, help="Input path (minhash output)")
    p.add_argument("--rows-per-band", type=int, default=4)


def run_clustering(config: PipelineConfig, **kwargs) -> dict:
    """Clustering & Dedup Step"""
    logger = PipelineLogger.get()
    manual_input = kwargs.get("input")
    if manual_input:
        input_path = Path(manual_input)
        _, output_dir = resolve_io_paths(config, "clustering")
    else:
        input_path, output_dir = resolve_io_paths(config, "clustering", "minhash")

    rows_per_band = getattr(config, "rows_per_band", kwargs.get("rows_per_band", 4))

    validate_input_path(input_path, "clustering")

    logger.info(f"Reading from {input_path}...")
    ds = rd.read_parquet(str(input_path.absolute()))

    limit = config.limit
    if limit > 0:
        logger.info(f"DEBUG: Limiting clustering input to {limit} records.")
        ds = ds.limit(limit)

    logger.info("Running ray_global_dedup logic...")
    result = ray_global_dedup(ds, rows_per_band=rows_per_band)

    keep_set = result["keep_set"]
    logger.info(f"Dedup finished. Kept {result['kept_docs']} / {result['total_docs']} docs.")

    ds_final = ds.map_batches(
        lambda batch: {k: [v[i] for i in range(len(v)) if batch["doc_id"][i] in keep_set] for k, v in batch.items()},
        batch_format="dict",
    )

    out_dir = output_dir / "deduped_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing deduped data to {out_dir}...")
    ds_final.write_parquet(str(out_dir))

    logger.info("Done.")

    return {
        "input_docs": result["total_docs"],
        "kept_docs": result["kept_docs"],
        "removed_docs": result["removed_docs"],
        "dedup_rate": result["dedup_rate"],
        "output_path": str(out_dir),
    }


def main():
    run_step_entrypoint(
        description="llm_data_pipeline.dedup.run_clustering",
        run_func=run_clustering,
        add_args_func=add_args,
        step_name="Clustering",
    )
