import argparse
import glob
import json
import os
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

# -----------------------------
# SentencePiece loader (per-process lazy init)
# -----------------------------
_SP = None


def _get_sp(model_path: str):
    global _SP
    if _SP is None:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        ok = sp.Load(model_path)
        if not ok:
            raise RuntimeError(f"Failed to load sentencepiece model: {model_path}")
        _SP = sp
    return _SP


def _resolve_eos_id(sp) -> int:
    # sentencepiece usually has eos_id() for </s>
    eos = sp.eos_id()
    if eos is not None and eos >= 0:
        return int(eos)

    # fallback to common piece
    for piece in ("</s>", "<eos>", "<EOS>"):
        pid = sp.piece_to_id(piece)
        if pid is not None and pid >= 0:
            return int(pid)

    raise RuntimeError("Cannot resolve EOS id from the sentencepiece model. Your model seems to have no EOS piece.")


# -----------------------------
# ConstantLengthDataset (streaming packer)
# -----------------------------
class ConstantLengthDataset:
    """
    Streamingly packs tokenized samples into fixed-length chunks.

    - Input: an iterable of token id lists (variable length)
    - Output: dict {"input_ids": list[int]} of fixed length seq_len

    Cross-block handling:
    - carry-over buffer splits samples across chunk boundaries (no dropping remainder)
    - optional EOS insertion per sample
    """

    def __init__(
        self,
        token_iter: Iterable[list[int]],
        seq_len: int,
        eos_id: int,
        add_eos: bool = True,
        ensure_eos: bool = True,
        drop_remainder: bool = True,
        buffer_flush_threshold: int = 1_000_000,
    ) -> None:
        self.token_iter = token_iter
        self.seq_len = int(seq_len)
        self.eos_id = int(eos_id)
        self.add_eos = bool(add_eos)
        self.ensure_eos = bool(ensure_eos)
        self.drop_remainder = bool(drop_remainder)

        # if buffer grows too big, we compact it (important for long streams)
        self.buffer_flush_threshold = int(buffer_flush_threshold)

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        buf: list[int] = []
        start = 0  # logical start index into buf (avoid O(n) pops)

        for ids in self.token_iter:
            if not ids:
                continue

            # Add/ensure EOS at sample boundary
            if self.add_eos:
                if self.ensure_eos:
                    if ids[-1] != self.eos_id:
                        ids = ids + [self.eos_id]
                else:
                    ids = ids + [self.eos_id]

            buf.extend(ids)

            # emit chunks as long as we have enough tokens
            while (len(buf) - start) >= self.seq_len:
                chunk = buf[start : start + self.seq_len]
                start += self.seq_len
                yield {"input_ids": chunk}

            # periodically compact buffer to avoid unbounded growth
            if start >= self.buffer_flush_threshold:
                buf = buf[start:]
                start = 0

        # handle tail
        remaining = len(buf) - start
        if remaining > 0 and not self.drop_remainder:
            # pad tail if you prefer; here we drop by default, but keep option
            tail = buf[start:]
            # If you want exactly seq_len always, you can pad with eos_id or 0
            # Here we pad with eos_id (common) for completeness.
            if len(tail) < self.seq_len:
                tail = tail + [self.eos_id] * (self.seq_len - len(tail))
            yield {"input_ids": tail}


# -----------------------------
# Parquet writer (fixed-size list[int32] column)
# -----------------------------
def write_parquet_shard(
    out_path: str,
    chunks: list[list[int]],
    seq_len: int,
    compression: str = "zstd",
) -> None:
    arr = np.asarray(chunks, dtype=np.int32)  # shape [n, L]
    if arr.ndim != 2 or arr.shape[1] != seq_len:
        raise ValueError(f"Bad chunk array shape: {arr.shape}, expected (*, {seq_len})")

    flat = pa.array(arr.reshape(-1), type=pa.int32())
    col = pa.FixedSizelistArray.from_arrays(flat, list_size=seq_len)  # pyright: ignore[reportAttributeAccessIssue]
    table = pa.Table.from_arrays([col], names=["input_ids"])
    pq.write_table(table, out_path, compression=compression)


# -----------------------------
# Pipeline
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm_model", type=str, default="outputs/dev/tokenizers/spm32k/spm_32k.model")
    ap.add_argument("--input_dir", type=str, default="outputs/dev/cleaned_parquet")
    ap.add_argument("--output_dir", type=str, default="outputs/dev/token_packing_parquet")

    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2))  # pyright: ignore[reportOptionalOperand]

    ap.add_argument("--batch_size", type=int, default=512, help="datasets.map batch size")
    ap.add_argument("--writer_batch_size", type=int, default=4096, help="datasets.map writer_batch_size")
    ap.add_argument("--shard_chunks", type=int, default=2048, help="packed chunks per output parquet shard")

    ap.add_argument("--add_eos", action="store_true", default=True)
    ap.add_argument("--no_add_eos", action="store_false", dest="add_eos")
    ap.add_argument("--ensure_eos", action="store_true", default=True)
    ap.add_argument("--no_ensure_eos", action="store_false", dest="ensure_eos")

    ap.add_argument("--drop_remainder", action="store_true", default=True)
    ap.add_argument("--keep_remainder", action="store_false", dest="drop_remainder")

    ap.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "none"])
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {args.input_dir}")

    # 1) Load parquet dataset (HF datasets)
    ds = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
    )
    if args.text_col not in ds.column_names:
        raise KeyError(f"Column '{args.text_col}' not found. Available: {ds.column_names}")

    # 2) Tokenize with datasets.map(num_proc=N)
    spm_model_path = args.spm_model

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        sp = _get_sp(spm_model_path)
        texts = batch[args.text_col]
        out_ids: list[list[int]] = []
        out_len: list[int] = []
        for t in texts:
            if t is None:
                out_ids.append([])
                out_len.append(0)
                continue
            s = str(t)
            if not s:
                out_ids.append([])
                out_len.append(0)
                continue
            ids = sp.encode(s, out_type=int)  # pyright: ignore[reportAttributeAccessIssue]
            out_ids.append(ids)
            out_len.append(len(ids))
        return {"input_ids": out_ids, "length": out_len}

    original_cols = ds.column_names
    ds_tok = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=original_cols,
        writer_batch_size=args.writer_batch_size,
        desc=f"Tokenizing with sentencepiece (num_proc={args.num_proc})",
    )

    # Resolve eos_id once (in main process)
    sp_main = _get_sp(spm_model_path)
    eos_id = _resolve_eos_id(sp_main)

    # 3) Stream pack: Concat -> Chunk to seq_len
    # Use ds_tok["input_ids"] as a Python iterable of lists (memory-mapped Arrow; OK)
    token_iter = (row for row in ds_tok["input_ids"])

    packer = ConstantLengthDataset(
        token_iter=token_iter,
        seq_len=args.seq_len,
        eos_id=eos_id,
        add_eos=args.add_eos,
        ensure_eos=args.ensure_eos,
        drop_remainder=args.drop_remainder,
    )

    # 4) Save to disk as parquet shards
    compression = None if args.compression == "none" else args.compression

    shard_idx = 0
    chunks_buf: list[list[int]] = []
    total_chunks = 0

    for item in packer:
        chunks_buf.append(item["input_ids"])
        if len(chunks_buf) >= args.shard_chunks:
            out_path = os.path.join(args.output_dir, f"packed_{shard_idx:05d}.parquet")
            write_parquet_shard(out_path, chunks_buf, seq_len=args.seq_len, compression=(compression or "none"))
            total_chunks += len(chunks_buf)
            chunks_buf.clear()
            shard_idx += 1

    if chunks_buf:
        out_path = os.path.join(args.output_dir, f"packed_{shard_idx:05d}.parquet")
        write_parquet_shard(out_path, chunks_buf, seq_len=args.seq_len, compression=(compression or "none"))
        total_chunks += len(chunks_buf)
        chunks_buf.clear()

    meta = {
        "seq_len": args.seq_len,
        "eos_id": eos_id,
        "spm_model": os.path.abspath(spm_model_path),
        "input_dir": os.path.abspath(args.input_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "num_proc": args.num_proc,
        "drop_remainder": args.drop_remainder,
        "add_eos": args.add_eos,
        "ensure_eos": args.ensure_eos,
        "total_chunks": total_chunks,
        "total_tokens_out": int(total_chunks) * int(args.seq_len),
    }
    with open(os.path.join(args.output_dir, "packing_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {total_chunks} packed chunks to: {args.output_dir}")
    print(f"[Meta] {os.path.join(args.output_dir, 'packing_meta.json')}")


if __name__ == "__main__":
    main()
