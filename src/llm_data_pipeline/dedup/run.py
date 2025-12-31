import argparse
from pathlib import Path

import ray.data as rd

from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    resolve_io_paths,
    run_step_entrypoint,
    validate_input_path,
    write_parquet,
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
        logger.info(f"Using manual input path: {input_path}")
    else:
        input_path, output_dir = resolve_io_paths(config, "clustering", "minhash")

    rows_per_band = getattr(config, "rows_per_band", kwargs.get("rows_per_band", 4))

    validate_input_path(input_path, "clustering")

    logger.info(f"Reading from {input_path}...")
    ds = rd.read_parquet(str(input_path.absolute()))
    initial_count = ds.count()
    logger.info(f"Found {initial_count} docs in input path before applying limit")

    # Check if dataset has required 'signature' column
    dataset_columns = ds.columns()
    if "signature" not in dataset_columns:
        if "minhash_sig" in dataset_columns:
            logger.warning(
                "Found 'minhash_sig' column but expected 'signature'. "
                "Renaming 'minhash_sig' to 'signature' for compatibility."
            )
            # Rename the column for compatibility
            ds = ds.rename_columns({"minhash_sig": "signature"})
        else:
            raise ValueError(
                f"Input dataset is missing required 'signature' column. "
                f"Available columns: {dataset_columns}. "
                f"Please ensure the 'minhash' step was run before 'clustering' step."
            )

    limit = config.limit
    if limit > 0:
        logger.info(f"DEBUG: Limiting clustering input to {limit} records.")
        ds = ds.limit(limit)
        limited_count = ds.count()
        logger.info(f"Input docs after limit: {limited_count}")

    logger.info("Running ray_global_dedup logic...")
    result = ray_global_dedup(ds, rows_per_band=rows_per_band)

    keep_set = result["keep_set"]
    logger.info(f"Dedup finished. Kept {result['kept_docs']} / {result['total_docs']} docs.")
    logger.info(f"Expected keep count from keep_set: {len(keep_set)}")

    logger.info(f"Filtering docs with keep_set size: {len(keep_set)}")

    # Use Ray Data's built-in filter method which is more reliable
    def filter_func(row):
        return row["doc_id"] in keep_set

    ds_final = ds.filter(filter_func)

    # Log dataset size after filtering
    final_count = ds_final.count()
    logger.info(f"Dataset size after filtering: {final_count}")
    if final_count != len(keep_set):
        logger.warning(f"Mismatch: keep_set size {len(keep_set)} vs actual kept docs {final_count}")

    out_dir = output_dir / "deduped_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing deduped data to {out_dir}...")
    # Use the write_parquet function from core.py
    write_parquet(ds_final, out_dir, logger)

    # Verify files were written
    import os

    if os.path.exists(out_dir):
        files = os.listdir(out_dir)
        logger.info(f"Found {len(files)} files in {out_dir}")
        for file in files[:5]:  # Show first 5 files
            logger.info(f"  - {file}")
    else:
        logger.error(f"Output directory {out_dir} does not exist!")

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
