import argparse
from pathlib import Path

import ray.data as rd

from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    run_step_entrypoint,
)
from llm_data_pipeline.dedup.dedup import ray_global_dedup


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", default=None, help="Input path (minhash output)")
    p.add_argument("--rows-per-band", type=int, default=4)


def _process_clustering(ds: rd.Dataset, config: PipelineConfig, **kwargs) -> rd.Dataset:
    """Core clustering processing function"""
    logger = PipelineLogger.get()
    
    rows_per_band = getattr(config, "rows_per_band", kwargs.get("rows_per_band", 4))
    
    # Check if dataset has required 'signature' column
    dataset_columns = ds.columns() or []
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

    logger.info("Running ray_global_dedup logic...")
    result = ray_global_dedup(ds, rows_per_band=rows_per_band)

    keep_set = result["keep_set"]
    logger.info(f"Dedup finished. Kept {result['kept_docs']} / {result['total_docs']} docs.")
    logger.info(f"Expected keep count from keep_set: {len(keep_set)}")

    logger.info(f"Dataset columns: {ds.columns()}")
    logger.info(f"Dataset sample: {ds.take(1)}")
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
    
    return ds_final


def run_clustering(config: PipelineConfig, **kwargs) -> dict:
    """Clustering & Dedup Step"""
    from llm_data_pipeline.core import step_wrapper
    logger = PipelineLogger.get()
    
    # Clustering supports manual input path override
    manual_input = kwargs.get("input")
    if manual_input:
        # If manual input is provided, we need to handle it specially
        # since step_wrapper expects input_step_name
        import time

        from llm_data_pipeline.core import (
            read_parquet,
            resolve_io_paths,
            validate_input_path,
            write_parquet,
        )
        
        total_start = time.time()
        input_path = Path(manual_input)
        _, output_dir = resolve_io_paths(config, "clustering")
        logger.info(f"Using manual input path: {input_path}")
        
        # Validate input path
        validate_input_path(input_path, "clustering")
        
        # Read data with stats
        ds, (input_file_count, input_total_size) = read_parquet(input_path, config)
        input_count = ds.count()
        
        # Core processing
        ds_out = _process_clustering(ds, config, **kwargs)
        
        # Write output with output stats
        output_path = output_dir / "deduped_parquet"
        output_file_count, output_total_size = write_parquet(ds_out, output_path, logger)
        output_count = ds_out.count()
        
        # Calculate total duration
        total_duration = time.time() - total_start
        
        # Prepare stats
        stats = {
            "step_name": "clustering",
            "input_path": str(input_path),
            "input_file_count": input_file_count,
            "input_total_size": input_total_size,
            "input_count": input_count,
            "output_path": str(output_path),
            "output_file_count": output_file_count,
            "output_total_size": output_total_size,
            "output_count": output_count,
            "duration_seconds": total_duration,
            "duration_human": time.strftime("%H:%M:%S", time.gmtime(total_duration)),
            "input_docs": input_count,
            "kept_docs": output_count,
            "removed_docs": input_count - output_count,
            "dedup_rate": (input_count - output_count) / input_count if input_count > 0 else 0.0,
        }
        
        return stats
    else:
        # Use standard step_wrapper for normal case
        stats = step_wrapper(
            step_name="clustering",
            process_func=_process_clustering,
            config=config,
            input_step_name="minhash",
            output_subdir="deduped_parquet",
            **kwargs
        )
        
        # Add clustering-specific stats
        stats.update({
            "input_docs": stats["input_count"],
            "kept_docs": stats["output_count"],
            "removed_docs": stats["input_count"] - stats["output_count"],
            "dedup_rate": (
                (stats["input_count"] - stats["output_count"]) / stats["input_count"] 
                if stats["input_count"] > 0 else 0.0
            ),
        })
        
        return stats


def main():
    run_step_entrypoint(
        description="llm_data_pipeline.dedup.run_clustering",
        run_func=run_clustering,
        add_args_func=add_args,
        step_name="Clustering",
    )
