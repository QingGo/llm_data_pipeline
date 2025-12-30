import argparse
from pathlib import Path

import ray
import ray.data as rd
import sentencepiece as spm
from transformers import AutoTokenizer


def write_shards_with_ray(
    parquet_dir: Path,
    out_dir: Path,
    num_shards: int,
    max_chars: int,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = rd.read_parquet(str(parquet_dir.absolute()))
    schema = ds.schema()
    assert schema is not None, "Dataset schema is None"
    if "text" not in schema.names:
        raise ValueError(f"'text' column not found. columns={schema.names}")

    ds = ds.select_columns(["text"]).filter(lambda r: r["text"] is not None and str(r["text"]).strip() != "")

    if max_chars > 0:
        ds = ds.map(lambda r: {"text": str(r["text"])[:max_chars]})

    # 把数据切成多个 shard，分别写文件，避免并发抢同一个文件句柄
    ds = ds.repartition(num_shards)
    splits = ds.split(num_shards, equal=True)

    @ray.remote
    def _write_one(i: int, split_ds, out_dir_str: str) -> str:
        p = Path(out_dir_str) / f"train_{i:04d}.txt"
        with p.open("w", encoding="utf-8") as f:
            for batch in split_ds.iter_batches(batch_size=4096):
                texts = batch["text"]
                for t in texts:
                    # 一行一个样本
                    s = str(t).replace("\n", " ").strip()
                    if s:
                        f.write(s + "\n")
        return str(p)

    paths = ray.get([_write_one.remote(i, s, str(out_dir.absolute())) for i, s in enumerate(splits)])
    return paths


def train_sentencepiece_py(
    input_txt_paths: list[str],
    model_prefix: Path,
    vocab_size: int,
    input_sentence_size: int,
    model_type: str,
    character_coverage: float,
):
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

    spm.SentencePieceTrainer.Train(**kwargs)


def compare_token_lengths(spm_model_path: Path, text: str):
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


def run_train_tokenizer(args) -> dict:
    """Train tokenizer step"""
    base_out = getattr(args, "output_dir", "./outputs/dev")
    input_path = Path(getattr(args, "parquet_dir", f"{base_out}/quality_parquet"))
    # In shell script: --work_dir outputs/dev/tokenizers/working
    # --model_prefix outputs/dev/tokenizers/my_spm

    # We defaults based on base_out if not set
    work_dir = Path(getattr(args, "work_dir", f"{base_out}/tokenizers/working"))
    model_prefix = Path(getattr(args, "model_prefix", f"{base_out}/tokenizers/my_spm"))

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist for training tokenizer.")

    shard_dir = work_dir / "corpus_txt"

    num_shards = getattr(args, "num_shards", 256)
    max_chars = getattr(args, "max_chars", 5000)
    vocab_size = getattr(args, "vocab_size", 32000)
    input_sentence_size = getattr(args, "input_sentence_size", 5_000_000)
    model_type = getattr(args, "model_type", "bpe")
    character_coverage = getattr(args, "character_coverage", 0.9995)
    sample_text = getattr(
        args, "sample_text", "我希望用同一段中文，比较自定义 tokenizer 与 Llama 3 tokenizer 的编码长度，并计算压缩率。"
    )

    print("Step 1) Write text shards from parquet...")
    txt_paths = write_shards_with_ray(
        parquet_dir=input_path,
        out_dir=shard_dir,
        num_shards=num_shards,
        max_chars=max_chars,
    )
    print(f"  wrote {len(txt_paths)} shard files into {shard_dir}")

    print("\nStep 2) Train SentencePiece...")
    train_sentencepiece_py(
        input_txt_paths=txt_paths,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        input_sentence_size=input_sentence_size,
        model_type=model_type,
        character_coverage=character_coverage,
    )

    spm_model_path = Path(str(model_prefix) + ".model")
    print("\nStep 3) Compare token lengths...")
    stats = compare_token_lengths(
        spm_model_path=spm_model_path,
        text=sample_text,
    )

    stats["model_path"] = str(spm_model_path)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", type=str, default="./outputs/dev/cleaned_parquet")
    ap.add_argument("--work_dir", type=str, default="./outputs/dev/tokenizers/spm32k_work")
    ap.add_argument(
        "--model_prefix",
        type=str,
        default="./outputs/dev/tokenizers/spm32k/spm_32k",
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
    args = ap.parse_args()

    # Note: main uses specific defaults, pipeline might override.
    ray.init(ignore_reinit_error=True)
    run_train_tokenizer(args)
