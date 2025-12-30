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


def run_clean(args) -> dict:
    """Pipeline entry point for cleaning"""
    # Resolve paths
    base_out = getattr(args, "output_dir", "./outputs/dev")
    input_path = Path(getattr(args, "input", f"{base_out}/ingest_parquet"))
    output_dir = Path(base_out)

    # Check input
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist for clean step.")

    ds = rd.read_parquet(str(input_path.absolute()))

    limit = getattr(args, "limit", 0)
    if limit > 0:
        print(f"DEBUG: Limiting clean input to {limit} records.")
        ds = ds.limit(limit)

    rules = CleanRules(
        min_chars=getattr(args, "min_chars", 200),
        max_chars=getattr(args, "max_chars", 200_000),
        min_non_ws_ratio=getattr(args, "min_non_ws_ratio", 0.7),
        min_alpha_cjk_ratio=getattr(args, "min_alpha_cjk_ratio", 0.4),
        max_punct_ratio=getattr(args, "max_punct_ratio", 0.25),
        max_dup_line_ratio=getattr(args, "max_dup_line_ratio", 0.35),
    )

    batch_size = getattr(args, "batch_size", 256)
    taskpool_size = getattr(args, "taskpool_size", 0)
    num_cpus = getattr(args, "num_cpus", 1.0)

    kept_ds, drop_ds = clean_dataset(
        ds,
        CleanConfig(batch_size=batch_size, rules=rules),
        taskpool_size=taskpool_size,
        num_cpus=num_cpus,
    )

    kept_dir = output_dir / "cleaned_parquet"
    drop_dir = output_dir / "dropped_parquet"
    kept_dir.mkdir(parents=True, exist_ok=True)
    drop_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing clean results to {kept_dir}...")
    kept_ds.write_parquet(str(kept_dir.absolute()))
    # Optional: write dropped? User said "specify which files to land".
    # For now assume we write both as before, or maybe skip dropped if not debug.
    # The original code wrote both. I'll keep writing both for safety/debug.
    drop_ds.write_parquet(str(drop_dir.absolute()))

    kept_count = kept_ds.count()
    drop_count = drop_ds.count()
    print("kept_count =", kept_count)
    print("drop_count =", drop_count)

    return {"input_count": ds.count(), "kept_count": kept_count, "drop_count": drop_count, "output_path": str(kept_dir)}


def main() -> None:
    """Cli wrapper"""
    args = parse_args()
    ray.init(address=args.ray_address or None)
    # Adapter for standalone run
    # args.input and args.output_dir are set by argparse
    run_clean(args)
