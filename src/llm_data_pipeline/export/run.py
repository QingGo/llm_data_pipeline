import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from llm_data_pipeline.core import (
    PipelineConfig,
    resolve_io_paths,
    run_step_entrypoint,
)

logger = logging.getLogger(__name__)


def add_args(p: argparse.ArgumentParser):
    p.add_argument("--input-dir", default=None, help="Packed parquet dir")
    p.add_argument("--output-file", default=None, help="Output binary file")
    p.add_argument("--dtype", default="uint16", choices=["uint16", "int32"])


def run_export(config: PipelineConfig, **kwargs) -> dict:
    """Export to binary step"""
    manual_input = kwargs.get("input_dir")
    base_calc_out = config.output_base

    if manual_input:
        input_dir = Path(manual_input)
    else:
        # resolve usually returns (input_dir, output_root)
        i_path, _ = resolve_io_paths(config, "export", "token_packing")
        input_dir = i_path

    manual_output = kwargs.get("output_file")
    if manual_output:
        output_file = Path(manual_output)
    else:
        output_file = base_calc_out / "final.bin"

    dtype_str = kwargs.get("dtype", "uint16")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir {input_dir} does not exist for export.")

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

    return {"files_processed": len(files), "total_tokens": total_tokens, "output_file": str(output_file)}


def main():
    run_step_entrypoint(
        description="llm_data_pipeline.export.run",
        run_func=run_export,
        add_args_func=add_args,
        step_name="Export",
        use_ray=False,
    )
