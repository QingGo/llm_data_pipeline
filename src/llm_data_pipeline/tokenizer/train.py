"""
Tokenizer Training Step.

This module handles the training of a SentencePiece tokenizer on the cleaned dataset.
It selects a subset of data, writes it to sharded text files, and runs the SentencePiece trainer.
It also includes utility functions to compare the trained tokenizer against reference models (e.g., Llama 3).
"""

import argparse
from pathlib import Path

import ray
import ray.data as rd
import sentencepiece as spm
from transformers import AutoTokenizer

from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    resolve_io_paths,
    run_step_entrypoint,
)


def write_shards_with_ray(
    parquet_dir: Path,
    out_dir: Path,
    num_shards: int,
    max_chars: int,
    limit: int = 0,
) -> tuple[list[str], int]:
    """
    Reads Parquet data and writes it to sharded text files for SentencePiece training.
    """
    logger = PipelineLogger.get()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = rd.read_parquet(str(parquet_dir.absolute()))
    schema = ds.schema()
    assert schema is not None, "Dataset schema is None"
    if "text" not in schema.names:
        raise ValueError(f"'text' column not found. columns={schema.names}")

    # Filter out None or empty text values
    ds = ds.select_columns(["text"]).filter(lambda r: r["text"] is not None and str(r["text"]).strip() != "")

    if limit > 0:
        logger.info(f"DEBUG: Limiting tokenizer training input to {limit} records.")
        ds = ds.limit(limit)

    # Get input count before further processing
    input_count = ds.count()
    logger.info(f"  Found {input_count} valid text records for training")

    if input_count == 0:
        raise ValueError("No valid text records found in the parquet files.")

    if max_chars > 0:
        ds = ds.map(lambda r: {"text": str(r["text"])[:max_chars]})

    # 确保分片数量不超过实际数据量，避免生成大量空文件
    actual_num_shards = min(num_shards, input_count)
    logger.info(f"  Using {actual_num_shards} shards (adjusted from {num_shards} based on data count)")

    # 把数据切成多个 shard，分别写文件，避免并发抢同一个文件句柄
    ds = ds.repartition(actual_num_shards)
    splits = ds.split(actual_num_shards, equal=True)

    @ray.remote
    def _write_one(i: int, split_ds, out_dir_str: str) -> tuple[str, int]:
        p = Path(out_dir_str) / f"train_{i:04d}.txt"
        written_lines = 0
        with p.open("w", encoding="utf-8") as f:
            for batch in split_ds.iter_batches(batch_size=4096):
                texts = batch["text"]
                for t in texts:
                    # 一行一个样本
                    s = str(t).replace("\n", " ").strip()
                    if s:
                        f.write(s + "\n")
                        written_lines += 1
        logger.info(f"  Wrote {written_lines} lines to shard {i:04d}")
        return str(p), written_lines

    results = ray.get([_write_one.remote(i, s, str(out_dir.absolute())) for i, s in enumerate(splits)])
    paths = [path for path, _ in results]
    return paths, input_count


def train_sentencepiece_py(
    input_txt_paths: list[str],
    model_prefix: Path,
    vocab_size: int,
    input_sentence_size: int,
    model_type: str,
    character_coverage: float,
):
    """
    Trains a SentencePiece model using the generated text shards.
    Redirects C++ output to Python logger.
    """
    import os
    import tempfile
    from contextlib import contextmanager

    from llm_data_pipeline.core import PipelineLogger

    logger = PipelineLogger.get()
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        input=",".join(input_txt_paths),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,  # unigram/bpe/char/word
        character_coverage=character_coverage,
        byte_fallback=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        remove_extra_whitespaces=True,
        normalization_rule_name="nmt_nfkc",
        num_threads=16,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )

    if input_sentence_size > 0:
        kwargs.update(
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
        )

    @contextmanager
    def redirect_cpp_output():
        """Redirect C++ stdout/stderr to temporary files using file descriptor manipulation."""
        # Create temporary files for stdout and stderr
        stdout_fd, stdout_path = tempfile.mkstemp()
        stderr_fd, stderr_path = tempfile.mkstemp()

        # Save original file descriptors
        original_stdout = os.dup(1)
        original_stderr = os.dup(2)

        try:
            # Redirect stdout and stderr to temporary files
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)

            # Close the temporary file descriptors (we'll read from the files later)
            os.close(stdout_fd)
            os.close(stderr_fd)

            yield stdout_path, stderr_path
        finally:
            # Restore original stdout and stderr
            os.dup2(original_stdout, 1)
            os.dup2(original_stderr, 2)

            # Close the saved file descriptors
            os.close(original_stdout)
            os.close(original_stderr)

    # Redirect stdout and stderr from sentencepiece C++ code to logger
    logger.info(f"Starting SentencePiece training with model_type={model_type}, vocab_size={vocab_size}")

    try:
        # Use file descriptor redirection to capture C++ output
        with redirect_cpp_output() as (stdout_path, stderr_path):
            # Run SentencePiece training
            spm.SentencePieceTrainer.Train(**kwargs)

        # Read captured output from temporary files
        with open(stdout_path) as f:
            stdout_output = f.read()
        with open(stderr_path) as f:
            stderr_output = f.read()

        # Clean up temporary files
        os.unlink(stdout_path)
        os.unlink(stderr_path)

        # Log captured output
        if stdout_output.strip():
            # Only log stdout if it's not just empty or whitespace
            logger.info(f"SentencePiece stdout: {stdout_output.strip()}")
        if stderr_output.strip():
            # Filter out all non-error logs from C++ code
            for line in stderr_output.splitlines():
                if line:
                    stripped_line = line.strip()
                    # Skip INFO logs
                    if "LOG(INFO)" in stripped_line and (".cc(" in stripped_line or ".cpp(" in stripped_line):
                        continue
                    # Skip all configuration logs
                    if any(
                        keyword in stripped_line
                        for keyword in [
                            "trainer_spec",
                            "input:",
                            "model_prefix:",
                            "vocab_size:",
                            "model_type:",
                            "character_coverage:",
                            "byte_fallback:",
                            "split_by_",
                            "remove_extra_whitespaces:",
                            "normalization_rule_name:",
                            "num_threads:",
                            "unk_id:",
                            "bos_id:",
                            "eos_id:",
                            "pad_id:",
                            "input_sentence_size:",
                            "shuffle_input_sentence:",
                            "input_format:",
                            "self_test_sample_size:",
                            "seed_sentencepiece_size:",
                            "shrinking_factor:",
                            "max_sentence_length:",
                            "num_sub_iterations:",
                            "max_sentencepiece_length:",
                            "split_digits:",
                            "pretokenization_delimiter:",
                            "treat_whitespace_as_suffix:",
                            "allow_whitespace_only_pieces:",
                            "required_chars:",
                            "vocabulary_output_piece_score:",
                            "train_extremely_large_corpus:",
                            "seed_sentencepieces_file:",
                            "hard_vocab_limit:",
                            "use_all_vocab:",
                            "unk_piece:",
                            "bos_piece:",
                            "eos_piece:",
                            "pad_piece:",
                            "unk_surface:",
                            "enable_differential_privacy:",
                            "differential_privacy_noise_level:",
                            "differential_privacy_clipping_threshold:",
                            "normalizer_spec",
                            "denormalizer_spec",
                            "}",
                        ]
                    ):
                        continue
                    # Skip meta piece logs
                    if "Adding meta_piece:" in stripped_line:
                        continue
                    # Skip empty lines and lines with just punctuation
                    if not stripped_line or stripped_line in ["{", "}", ":"]:
                        continue
                    # Only log actual warnings and errors (lines containing .cc( and LOG(WARNING) or LOG(ERROR))
                    has_cc_file = ".cc(" in stripped_line or ".cpp(" in stripped_line
                    has_log = "LOG(WARNING)" in stripped_line or "LOG(ERROR)" in stripped_line
                    if has_cc_file and has_log:
                        logger.warning(f"SentencePiece stderr: {stripped_line}")
    except Exception as e:
        logger.error(f"SentencePiece training failed: {e}")
        raise

    logger.info("SentencePiece training completed successfully")


def compare_token_lengths(spm_model_path: Path, text: str):
    """
    Compares the trained SentencePiece model against a reference tokenizer (Llama 3).
    Calculates compression ratio and token savings on a sample text.
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(str(spm_model_path))
    ids_spm = sp.EncodeAsIds(text)

    # Llama 3 tokenizer 是 tiktoken BPE，下载命令
    # modelscope download --model LLM-Research/Meta-Llama-3-8B \
    # special_tokens_map.json tokenizer.json tokenizer_config.json --local_dir ./llama3_tokenizer
    tok_llama3 = AutoTokenizer.from_pretrained(
        "./llama3_tokenizer",
        local_files_only=True,
        use_fast=True,
    )
    ids_llama3 = tok_llama3.encode(text, add_special_tokens=False)

    n_spm = len(ids_spm)
    n_llama3 = len(ids_llama3)

    # “压缩率”给你两个口径：
    # 1) ratio = n_llama3 / n_spm：>1 表示你更短（相对 Llama3 压缩倍数）
    # 2) saving = 1 - n_spm / n_llama3：正数表示你更省 token（节省比例）
    ratio = (n_llama3 / n_spm) if n_spm > 0 else float("inf")
    saving = (1.0 - (n_spm / n_llama3)) if n_llama3 > 0 else 0.0

    print("\n=== Compare on the same Chinese text ===")
    print("Text:", text)
    print(f"Custom SentencePiece tokens: {n_spm}")
    print(f"Llama 3 tokens:             {n_llama3}")
    print(f"Compression ratio (Llama3 / Custom): {ratio:.4f}x")
    print(f"Token saving vs Llama3:             {saving * 100:.2f}%")

    return {
        "custom_spm_tokens": n_spm,
        "llama3_tokens": n_llama3,
        "compression_ratio_llama3_over_custom": ratio,
        "token_saving_vs_llama3": saving,
    }


def run_train_tokenizer(config: PipelineConfig, **kwargs) -> dict:
    """Train tokenizer step"""
    logger = PipelineLogger.get()
    import time

    total_start = time.time()

    input_path_base, output_dir_base = resolve_io_paths(config, "train_tokenizer", "clustering")

    def get_arg(name, default=None):
        return kwargs.get(name, default)

    # Path resolution
    parquet_dir_arg = get_arg("parquet_dir")
    if parquet_dir_arg:
        input_path = Path(parquet_dir_arg)
    elif input_path_base:
        # resolve_io_paths returns folder
        input_path = input_path_base
    else:
        input_path = output_dir_base / "deduped_parquet"

    # We defaults based on base_out if not set
    work_dir = Path(get_arg("work_dir", output_dir_base / "tokenizers/working"))
    model_prefix = Path(get_arg("model_prefix", output_dir_base / "tokenizers/my_spm"))

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist for training tokenizer.")

    shard_dir = work_dir / "corpus_txt"

    num_shards = int(get_arg("num_shards", 256))
    max_chars = int(get_arg("max_chars", 5000))
    vocab_size = int(get_arg("vocab_size", 32000))
    input_sentence_size = int(get_arg("input_sentence_size", 5_000_000))
    model_type = get_arg("model_type", "bpe")
    character_coverage = float(get_arg("character_coverage", 0.9995))
    sample_text = get_arg(
        "sample_text", "我希望用同一段中文，比较自定义 tokenizer 与 Llama 3 tokenizer 的编码长度，并计算压缩率。"
    )

    print("Step 1) Write text shards from parquet...")
    process_start = time.time()
    txt_paths, input_count = write_shards_with_ray(
        parquet_dir=input_path,
        out_dir=shard_dir,
        num_shards=num_shards,
        max_chars=max_chars,
        limit=int(get_arg("limit", 0)),
    )
    logger.info(f"  wrote {len(txt_paths)} shard files into {shard_dir}")

    # Validate that we have non-empty text files for training
    non_empty_txt_paths = []
    for txt_path in txt_paths:
        try:
            if Path(txt_path).stat().st_size > 0:
                non_empty_txt_paths.append(txt_path)
        except FileNotFoundError:
            logger.warning(f"Shard file not found: {txt_path}")

    if not non_empty_txt_paths:
        raise ValueError(f"No non-empty text shard files found. All {len(txt_paths)} shard files are empty.")

    logger.info(f"  Found {len(non_empty_txt_paths)} non-empty shard files for training")

    logger.info("Step 2) Train SentencePiece...")
    train_sentencepiece_py(
        input_txt_paths=non_empty_txt_paths,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        input_sentence_size=input_sentence_size,
        model_type=model_type,
        character_coverage=character_coverage,
    )
    process_end = time.time()

    spm_model_path = Path(str(model_prefix) + ".model")
    logger.info("Step 3) Compare token lengths...")
    token_stats = compare_token_lengths(
        spm_model_path=spm_model_path,
        text=sample_text,
    )
    total_end = time.time()

    # Get input and output stats
    from llm_data_pipeline.core import get_directory_stats

    input_file_count, input_total_size = get_directory_stats(input_path)
    output_file_count = 2  # model and vocab file

    # Calculate output total size
    model_size = spm_model_path.stat().st_size if spm_model_path.exists() else 0
    vocab_path = Path(str(model_prefix) + ".vocab")
    vocab_size_bytes = vocab_path.stat().st_size if vocab_path.exists() else 0
    output_total_size = model_size + vocab_size_bytes

    # Prepare comprehensive stats in the same format as other steps
    stats = {
        "step_name": "train_tokenizer",
        "input_path": str(input_path),
        "input_file_count": input_file_count,
        "input_total_size": input_total_size,
        "input_count": input_count,  # Use the actual input count from dataset
        "output_path": str(model_prefix.parent),
        "output_file_count": output_file_count,
        "output_total_size": output_total_size,
        "output_count": 0,  # Train tokenizer doesn't produce output records
        "model_path": str(spm_model_path),
        "duration_seconds": total_end - total_start,
        "duration_human": time.strftime("%H:%M:%S", time.gmtime(total_end - total_start)),
        "process_duration_seconds": process_end - process_start,
        # Add tokenizer-specific stats
        **token_stats,
    }

    return stats


def add_args(ap: argparse.ArgumentParser):
    """
    Adds Tokenizer Training step specific arguments.
    """
    ap.add_argument("--parquet_dir", type=str, default=None)
    ap.add_argument("--work_dir", type=str, default=None)
    ap.add_argument(
        "--model_prefix",
        type=str,
        default=None,
    )
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--num_shards", type=int, default=256)
    ap.add_argument("--max_chars", type=int, default=5000)
    ap.add_argument("--input_sentence_size", type=int, default=5_000_000)  # 0 表示全量
    ap.add_argument(
        "--model_type",
        type=str,
        default="bpe",
        choices=["unigram", "bpe", "char", "word"],
    )
    ap.add_argument("--character_coverage", type=float, default=0.9995)
    ap.add_argument(
        "--sample_text",
        type=str,
        default="我希望用同一段中文，比较自定义 tokenizer 与 Llama 3 tokenizer 的编码长度，并计算压缩率。",
    )


def main():
    run_step_entrypoint(
        description="Tokenizer Training",
        run_func=run_train_tokenizer,
        add_args_func=add_args,
        step_name="TrainTokenizer",
    )
