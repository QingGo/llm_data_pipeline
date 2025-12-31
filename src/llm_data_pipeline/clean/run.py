"""清洗阶段运行入口：读取 ingest 结果并按规则过滤输出"""

import argparse

import ray.data as rd

from llm_data_pipeline.clean.rules import CleanRules
from llm_data_pipeline.clean.step import CleanConfig, clean_dataset
from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    resolve_io_paths,
    run_step_entrypoint,
    validate_input_path,
    write_parquet,
)


def add_args(p: argparse.ArgumentParser) -> None:
    """添加 Clean 特有参数"""
    # rules
    p.add_argument("--taskpool-size", type=int, default=0)
    p.add_argument("--num-cpus", type=float, default=1.0)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=200_000)
    p.add_argument("--min-non-ws-ratio", type=float, default=0.7)
    p.add_argument("--min-alpha-cjk-ratio", type=float, default=0.4)
    p.add_argument("--max-punct-ratio", type=float, default=0.25)
    p.add_argument("--max-dup-line-ratio", type=float, default=0.35)


def run_clean(config: PipelineConfig, **kwargs) -> dict:
    """Pipeline entry point for cleaning"""
    logger = PipelineLogger.get()

    # Resolve paths
    input_path, output_dir = resolve_io_paths(config, "clean", "ingest")

    # Validate input path
    validate_input_path(input_path, "clean")

    ds = rd.read_parquet(str(input_path.absolute()))

    limit = config.limit
    if limit > 0:
        logger.info(f"DEBUG: Limiting clean input to {limit} records.")
        ds = ds.limit(limit)

    rules = CleanRules(
        min_chars=kwargs.get("min_chars", 200),
        max_chars=kwargs.get("max_chars", 200_000),
        min_non_ws_ratio=kwargs.get("min_non_ws_ratio", 0.7),
        min_alpha_cjk_ratio=kwargs.get("min_alpha_cjk_ratio", 0.4),
        max_punct_ratio=kwargs.get("max_punct_ratio", 0.25),
        max_dup_line_ratio=kwargs.get("max_dup_line_ratio", 0.35),
    )

    batch_size = config.batch_size
    taskpool_size = kwargs.get("taskpool_size", 0)
    num_cpus = kwargs.get("num_cpus", 1.0)

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

    write_parquet(kept_ds, kept_dir, logger)

    kept_count = kept_ds.count()
    drop_count = drop_ds.count()
    logger.info(f"kept_count = {kept_count}")
    logger.info(f"drop_count = {drop_count}")

    return {"input_count": ds.count(), "kept_count": kept_count, "drop_count": drop_count, "output_path": str(kept_dir)}


def main() -> None:
    run_step_entrypoint(
        description="llm_data_pipeline.clean.run",
        run_func=run_clean,
        add_args_func=add_args,
        step_name="Clean",
    )
