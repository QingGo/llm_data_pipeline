import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    get_directory_stats,
    resolve_io_paths,
    run_step_entrypoint,
    validate_input_path,
)


def add_args(p: argparse.ArgumentParser):
    p.add_argument("--input-dir", default=None, help="Packed parquet dir")
    p.add_argument("--output-file", default=None, help="Output binary file")
    p.add_argument("--dtype", default="uint16", choices=["uint16", "int32"])


def run_export(config: PipelineConfig, **kwargs) -> dict:
    """Export to binary step"""
    logger = PipelineLogger.get()
    base_calc_out = config.output_base
    import time
    total_start = time.time()

    # Resolve input directory
    input_dir = kwargs.get("input_dir")
    if input_dir:
        input_dir = Path(input_dir)
    else:
        # resolve usually returns (input_dir, output_root)
        i_path, _ = resolve_io_paths(config, "export", "token_packing")
        input_dir = i_path

    # Resolve output file
    output_file = kwargs.get("output_file")
    if output_file:
        output_file = Path(output_file)
    else:
        output_file = base_calc_out / "final.bin"

    dtype_str = kwargs.get("dtype", "uint16")

    # Validate input path
    validate_input_path(input_dir, "export")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        logger.warning(f"No files found in {input_dir}")
        return {"status": "skipped"}

    logger.info(f"Exporting {len(files)} files to {output_file} as {dtype_str}...")

    dtype = np.uint16 if dtype_str == "uint16" else np.int32

    total_tokens = 0
    limit = config.limit
    files_processed = 0

    # Get input stats
    input_file_count, input_total_size = get_directory_stats(input_dir)

    # Calculate output_count (token chunks) by reading the first file's schema
    output_count = 0
    if files:
        first_file = files[0]
        table = pq.read_table(first_file, columns=["input_ids"])
        # Assume all files have the same chunk size
        chunks_per_file = table.num_rows
        output_count = chunks_per_file * len(files)

    with open(output_file, "wb") as f:
        for pf in files:
            # Check file limit if strictly needed, though limit usually refers to records.
            # Here it's ambiguous. pipeline usually sets limit for records.
            # But let's assume if limit is extremely small (like 1-5), it might mean files for debug?
            # Or huge for records?
            # Existing code:
            # if limit > 0 and files_processed >= limit:
            # This interprets limit as files limit for export. Consistent with old code.

            if limit > 0 and files_processed >= limit:
                logger.info(f"DEBUG: Reached file limit {limit} for export. Stopping.")
                break

            files_processed += 1

            table = pq.read_table(pf, columns=["input_ids"])
            # table has one column "input_ids" which is FixedSizeList or List
            # We flatten it.
            # PyArrow table -> pandas or numpy
            # Flattening FixedSizeList in pyarrow:
            col = table["input_ids"]
            # col is ChunkedArray of FixedSizeListArray
            # We can get values which is the flat array
            # But we must be careful if it's chunked.

            for chunk in col.chunks:
                # chunk is FixedSizeListArray
                # chunk.values is the flat underlying array (if no nulls)
                # We assume no nulls from packing step.
                flat_values = chunk.values.to_numpy()

                # Check bounds if uint16
                if dtype_str == "uint16":
                    if flat_values.max() >= 65535:
                        logger.warning("WARNING: Token ID > 65535 found, but exporting as uint16!")

                data = flat_values.astype(dtype)
                f.write(data.tobytes())
                total_tokens += len(data)

    logger.info(f"Done. Wrote {total_tokens} tokens to {output_file}")

    # Get output stats
    output_file_count = 1  # Only one output file
    output_total_size = output_file.stat().st_size if output_file.exists() else 0

    # Calculate total duration
    total_end = time.time()
    total_duration = total_end - total_start

    # Prepare comprehensive stats in the same format as other steps
    stats = {
        "step_name": "export",
        "input_path": str(input_dir),
        "input_file_count": input_file_count,
        "input_total_size": input_total_size,
        "input_count": output_count,  # Number of token chunks
        "output_path": str(output_file.parent),
        "output_file_count": output_file_count,
        "output_total_size": output_total_size,
        "output_count": output_count,  # Number of token chunks
        "files_processed": len(files),
        "total_tokens": total_tokens,
        "output_file": str(output_file),
        "dtype": dtype_str,
        "duration_seconds": total_duration,
        "duration_human": time.strftime("%H:%M:%S", time.gmtime(total_duration)),
        "process_duration_seconds": total_duration,  # Same as total for now
    }

    return stats


def main():
    run_step_entrypoint(
        description="llm_data_pipeline.export.run",
        run_func=run_export,
        add_args_func=add_args,
        step_name="Export",
        use_ray=False,
    )
