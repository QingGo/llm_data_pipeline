"""清洗阶段运行入口：读取 ingest 结果并按规则过滤输出"""

import argparse
from pathlib import Path

import ray
import ray.data as rd

from llm_data_pipeline.clean.rules import CleanRules
from llm_data_pipeline.clean.step import CleanConfig, clean_dataset


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    p = argparse.ArgumentParser("llm_data_pipeline.clean.run")
    p.add_argument(
        "--input",
        default="./outputs/dev/ingest_parquet",
        help="e.g. outputs/dev/ingest_parquet",
    )
    p.add_argument("--output-dir", default="./outputs/dev", help="e.g. outputs/dev")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ray-address", default=None)
    p.add_argument("--taskpool-size", type=int, default=0)
    p.add_argument("--num-cpus", type=float, default=1.0)
    # rules
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=200_000)
    p.add_argument("--min-non-ws-ratio", type=float, default=0.7)
    p.add_argument("--min-alpha-cjk-ratio", type=float, default=0.4)
    p.add_argument("--max-punct-ratio", type=float, default=0.25)
    p.add_argument("--max-dup-line-ratio", type=float, default=0.35)
    return p.parse_args()


def main() -> None:
    """入口函数：初始化 Ray，读取数据，执行清洗并写出结果"""
    args = parse_args()
    ray.init(address=args.ray_address or None)

    ds = rd.read_parquet(args.input)

    rules = CleanRules(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_non_ws_ratio=args.min_non_ws_ratio,
        min_alpha_cjk_ratio=args.min_alpha_cjk_ratio,
        max_punct_ratio=args.max_punct_ratio,
        max_dup_line_ratio=args.max_dup_line_ratio,
    )

    kept_ds, drop_ds = clean_dataset(
        ds,
        CleanConfig(batch_size=args.batch_size, rules=rules),
        taskpool_size=args.taskpool_size,
        num_cpus=args.num_cpus,
    )

    out = Path(args.output_dir)
    kept_dir = out / "cleaned_parquet"
    drop_dir = out / "dropped_parquet"
    kept_dir.mkdir(parents=True, exist_ok=True)
    drop_dir.mkdir(parents=True, exist_ok=True)

    kept_ds.write_parquet(str(kept_dir))
    drop_ds.write_parquet(str(drop_dir))

    print("kept_count =", kept_ds.count())
    print("drop_count =", drop_ds.count())


if __name__ == "__main__":
    main()
