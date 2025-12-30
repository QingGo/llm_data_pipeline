"""MinHash 计算运行入口：读取清洗后的数据，计算 MinHash Signature 并保存"""

import argparse
from pathlib import Path
from typing import Any

import ray
import ray.data as rd

from llm_data_pipeline.dedup.minhash import char_ngrams, datasketch_minhash


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("llm_data_pipeline.dedup.run_minhash")
    p.add_argument(
        "--input",
        default="./outputs/dev/cleaned_parquet",
        help="Input path (cleaned data)",
    )
    p.add_argument(
        "--output-dir",
        default="./outputs/dev",
        help="Output parent directory",
    )
    p.add_argument("--num-perm", type=int, default=128, help="MinHash num permutations")
    p.add_argument("--ngram", type=int, default=5, help="N-gram size for shingling")
    p.add_argument("--ray-address", default=None)
    return p.parse_args()


class MinHashMapper:
    def __init__(self, num_perm: int, ngram: int):
        self.num_perm = num_perm
        self.ngram = ngram

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        text = row.get("text", "")
        # 如果 text 为 None 或非字符串，需做容错
        if not isinstance(text, str):
            text = ""

        # 1. 生成 shingles
        shingles = char_ngrams(text, n=self.ngram)

        # 2. 计算 MinHash
        mh = datasketch_minhash(shingles, k=self.num_perm)

        # 3. 保存签名 (转为 list 以便存储)
        # datasketch.MinHash.hashvalues 是 numpy array
        row["minhash_sig"] = mh.hashvalues.tolist()

        return row


def main():
    args = parse_args()
    print("Initializing Ray...")
    ray.init(address=args.ray_address, ignore_reinit_error=True)
    print("Ray initialized.")

    print(f"Reading from {args.input}...")
    try:
        ds = rd.read_parquet(args.input)
    except Exception as e:
        print(f"Error reading input: {e}")
        return

    mapper = MinHashMapper(num_perm=args.num_perm, ngram=args.ngram)

    # 使用 map 对每行数据计算签名
    ds_with_sig = ds.map(mapper)

    out_dir = Path(args.output_dir) / "minhash_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {out_dir}...")
    ds_with_sig.write_parquet(str(out_dir))

    print("Done.")


if __name__ == "__main__":
    main()
