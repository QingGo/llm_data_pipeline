import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser("llm_data_pipeline.export.run")
    p.add_argument("--input-dir", default="./outputs/dev/token_packing_parquet", help="Packed parquet dir")
    p.add_argument("--output-file", default="./outputs/dev/final_tokens.bin", help="Output binary file")
    p.add_argument("--dtype", default="uint16", choices=["uint16", "int32"])
    return p.parse_args()


def run_export(args) -> dict:
    """Export to binary step"""
    base_out = getattr(args, "output_dir", "./outputs/dev")
    input_dir = Path(getattr(args, "input_dir", f"{base_out}/token_packing_parquet"))
    output_file = Path(getattr(args, "output_file", f"{base_out}/final.bin"))
    dtype_str = getattr(args, "dtype", "uint16")

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input dir {input_dir} does not exist for export."
        )  # Prone to error if previous step skipped.

    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        print(f"No files found in {input_dir}")
        return {"status": "skipped"}

    print(f"Exporting {len(files)} files to {output_file} as {dtype_str}...")

    dtype = np.uint16 if dtype_str == "uint16" else np.int32

    total_tokens = 0
    limit = getattr(args, "limit", 0)
    files_processed = 0

    with open(output_file, "wb") as f:
        for pf in files:
            if limit > 0 and files_processed >= limit:
                print(f"DEBUG: Reached file limit {limit} for export. Stopping.")
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
                        print("WARNING: Token ID > 65535 found, but exporting as uint16!")

                data = flat_values.astype(dtype)
                f.write(data.tobytes())
                total_tokens += len(data)

    print(f"Done. Wrote {total_tokens} tokens to {output_file}")

    return {"files_processed": len(files), "total_tokens": total_tokens, "output_file": str(output_file)}


def main():
    args = parse_args()
    run_export(args)
