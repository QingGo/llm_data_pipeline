import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import datasets

# 每个 worker 进程内各自 lazy-init（避免 tokenizer 对象跨进程 pickle）
_HF_TOKENIZER = None
_SPM = None


def _get_hf_tokenizer(name_or_path: str):
    global _HF_TOKENIZER
    if _HF_TOKENIZER is None:
        from transformers import AutoTokenizer

        _HF_TOKENIZER = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        # 某些 tokenizer 没有 pad_token，会在某些调用路径里报错；这里兜底
        if _HF_TOKENIZER.pad_token is None and _HF_TOKENIZER.eos_token is not None:
            _HF_TOKENIZER.pad_token = _HF_TOKENIZER.eos_token
    return _HF_TOKENIZER


def _get_spm(model_path: str):
    global _SPM
    if _SPM is None:
        import sentencepiece as spm

        _SPM = spm.SentencePieceProcessor()
        # 用稳定的 C++ 绑定 API（比 kwargs 构造更不容易被类型检查器误伤）
        _SPM.Load(model_path)
    return _SPM


def _tokenize_batch(
    batch: dict[str, list[Any]],
    *,
    text_col: str,
    hf_tokenizer: str | None,
    spm_model: str | None,
    add_special_tokens: bool,
) -> dict[str, Any]:
    texts = batch[text_col]

    if hf_tokenizer:
        tok = _get_hf_tokenizer(hf_tokenizer)
        enc = tok(
            texts,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
        )
        input_ids = enc["input_ids"]
    else:
        sp = _get_spm(spm_model)  # type: ignore[arg-type]
        input_ids = [sp.EncodeAsIds(t) for t in texts]

    num_tokens = [len(x) for x in input_ids]
    return {"input_ids": input_ids, "num_tokens": num_tokens}


def _sum_tokens_fast(ds: datasets.Dataset) -> int:
    # 尽量用 Arrow 侧聚合，避免把整列拉回 Python
    try:
        import pyarrow.compute as pc

        col = ds.data.column("num_tokens")
        return int(pc.sum(col).as_py())  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        return int(sum(ds["num_tokens"]))  # fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/dev/cleaned_parquet")
    parser.add_argument("--output-dir", default="outputs/dev/token_parquet")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--num-proc", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--writer-batch-size", type=int, default=2000)
    parser.add_argument("--num-shards", type=int, default=0, help="0 表示默认=输入 parquet 文件数（至少 1）")
    parser.add_argument("--keep-text", action="store_true", help="默认会移除 text 列，只保留 input_ids/num_tokens")
    parser.add_argument("--add-special-tokens", action="store_true")

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--hf-tokenizer", default=None, help="例如本地 llama3 tokenizer 目录 或 HF 模型名")
    g.add_argument("--spm-model", default=None, help="例如 outputs/dev/spm_32k.model")

    parser.add_argument("--compression", default="zstd", help="parquet 压缩：zstd/snappy/gzip/none")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under: {in_dir}")

    # 读取 parquet（datasets 支持 parquet loader）
    data_files = [str(p) for p in files]
    ds = datasets.load_dataset("parquet", data_files=data_files, split="train")

    if args.text_col not in ds.column_names:
        raise ValueError(f"'{args.text_col}' column not found. columns={ds.column_names}")

    remove_cols = None
    if not args.keep_text:
        remove_cols = [c for c in ds.column_names if c != args.text_col]

    # 分词（CPU 多进程：num_proc）:contentReference[oaicite:1]{index=1}
    t0 = time.perf_counter()
    ds_tok = ds.map(
        lambda batch: _tokenize_batch(
            batch,
            text_col=args.text_col,
            hf_tokenizer=args.hf_tokenizer,
            spm_model=args.spm_model,
            add_special_tokens=args.add_special_tokens,
        ),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=remove_cols,
        writer_batch_size=args.writer_batch_size,
        load_from_cache_file=False,
        desc="Tokenizing",
    )
    t1 = time.perf_counter()

    total_tokens = _sum_tokens_fast(ds_tok)
    map_sec = t1 - t0
    tok_per_sec = total_tokens / map_sec if map_sec > 0 else 0.0

    print(f"[tokenize] rows={ds_tok.num_rows:,}")
    print(f"[tokenize] tokens={total_tokens:,}")
    print(f"[tokenize] time={map_sec:.2f}s")
    print(f"[tokenize] throughput={tok_per_sec:,.0f} tok/s  (~{tok_per_sec / 1e5:.2f}×10^5 tok/s)")

    # 写 parquet（datasets 支持导出 parquet）:contentReference[oaicite:2]{index=2}
    num_shards = args.num_shards if args.num_shards > 0 else max(1, len(files))

    w0 = time.perf_counter()
    total_bytes = 0
    for i in range(num_shards):
        shard = ds_tok.shard(num_shards=num_shards, index=i, contiguous=True)
        out_path = out_dir / f"part-{i:05d}.parquet"
        writer_kwargs = {}
        if args.compression.lower() != "none":
            writer_kwargs["compression"] = args.compression  # 例如 "zstd" / "snappy"

        written = shard.to_parquet(str(out_path), **writer_kwargs)
        total_bytes += int(written)
    w1 = time.perf_counter()

    write_sec = w1 - w0
    print(f"[write] shards={num_shards} time={write_sec:.2f}s bytes={total_bytes:,}")

    metrics = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "num_input_files": len(files),
        "rows": ds_tok.num_rows,
        "tokens": total_tokens,
        "num_proc": args.num_proc,
        "batch_size": args.batch_size,
        "writer_batch_size": args.writer_batch_size,
        "tokenizer": {
            "hf_tokenizer": args.hf_tokenizer,
            "spm_model": args.spm_model,
            "add_special_tokens": args.add_special_tokens,
        },
        "time_sec": {"tokenize_map": map_sec, "write_parquet": write_sec},
        "throughput_tok_per_sec": tok_per_sec,
        "parquet": {"compression": args.compression, "num_shards": num_shards, "bytes_written": total_bytes},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] metrics -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    # 重要：多进程下建议总是加 main guard（macOS/Windows 默认 spawn）
    main()


"""
uv run python src/llm_data_pipeline/tokenizer/compare.py \
  --spm-model outputs/dev/tokenizers/spm32k/spm_32k.model \
  --input-dir outputs/dev/cleaned_parquet \
  --output-dir outputs/dev/token_parquet \
  --num-proc 8 \
  --batch-size 2000

Tokenizing (num_proc=8): 107556 examples [01:14, 718.15 examples/s]                                
[tokenize] rows=53,778
[tokenize] tokens=125,510,712
[tokenize] time=74.94s
[tokenize] throughput=1,674,775 tok/s  (~16.75×10^5 tok/s)
[write] shards=140 time=11.46s bytes=949,550,178
[done] metrics -> outputs/dev/token_parquet/metrics.json

uv run python src/llm_data_pipeline/tokenizer/compare.py \
  --hf-tokenizer llama3_tokenizer \
  --input-dir outputs/dev/cleaned_parquet \
  --output-dir outputs/dev/token_parquet \
  --num-proc 8 \
  --batch-size 2000

Tokenizing (num_proc=8): 100%|███████████████████████| 53778/53778 [02:00<00:00, 444.48 examples/s]
[tokenize] rows=53,778
[tokenize] tokens=129,409,748
[tokenize] time=121.06s
[tokenize] throughput=1,068,954 tok/s  (~10.69×10^5 tok/s)
[write] shards=140 time=17.25s bytes=965,146,322
[done] metrics -> outputs/dev/token_parquet/metrics.json
"""
