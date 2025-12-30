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


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        print(f"No files found in {input_dir}")
        return

    print(f"Exporting {len(files)} files to {output_file} as {args.dtype}...")

    dtype = np.uint16 if args.dtype == "uint16" else np.int32

    total_tokens = 0
    with open(output_file, "wb") as f:
        for pf in files:
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
                if args.dtype == "uint16":
                    if flat_values.max() >= 65535:
                        print("WARNING: Token ID > 65535 found, but exporting as uint16!")

                data = flat_values.astype(dtype)
                f.write(data.tobytes())
                total_tokens += len(data)

    print(f"Done. Wrote {total_tokens} tokens to {output_file}")


if __name__ == "__main__":
    main()
