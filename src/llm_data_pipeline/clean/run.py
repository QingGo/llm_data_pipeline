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


def _process_clean(ds: rd.Dataset, config: PipelineConfig, **kwargs) -> tuple[rd.Dataset, rd.Dataset]:
    """Core cleaning processing function"""
    logger = PipelineLogger.get()
    
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
    
    return kept_ds, drop_ds


def run_clean(config: PipelineConfig, **kwargs) -> dict:
    """Pipeline entry point for cleaning"""
    from llm_data_pipeline.core import step_wrapper, read_parquet, resolve_io_paths, validate_input_path, write_parquet
    logger = PipelineLogger.get()
    import time
    
    total_start = time.time()
    
    # Clean step is special because it produces two outputs: kept and dropped
    # We'll use a custom approach that leverages the step_wrapper pattern but handles both outputs
    
    # Resolve paths
    input_path, output_dir = resolve_io_paths(config, "clean", "ingest")
    
    # Validate input path
    validate_input_path(input_path, "clean")
    
    # Read data with stats
    ds, (input_file_count, input_total_size) = read_parquet(input_path, config)
    input_count = ds.count()
    
    # Core processing
    kept_ds, drop_ds = _process_clean(ds, config, **kwargs)
    
    # Write kept and dropped datasets
    kept_dir = output_dir / "cleaned_parquet"
    drop_dir = output_dir / "dropped_parquet"
    
    # Write kept dataset
    kept_file_count, kept_total_size = write_parquet(kept_ds, kept_dir, logger)
    kept_count = kept_ds.count()
    
    # Write dropped dataset
    write_parquet(drop_ds, drop_dir, logger)
    drop_count = drop_ds.count()
    
    # Calculate total duration
    total_duration = time.time() - total_start
    
    stats = {
        "step_name": "clean",
        "input_path": str(input_path),
        "input_file_count": input_file_count,
        "input_total_size": input_total_size,
        "input_count": input_count,
        "output_path": str(kept_dir),
        "output_file_count": kept_file_count,
        "output_total_size": kept_total_size,
        "output_count": kept_count,
        "kept_count": kept_count,
        "drop_count": drop_count,
        "duration_seconds": total_duration,
        "duration_human": time.strftime("%H:%M:%S", time.gmtime(total_duration)),
    }
    
    logger.info(f"Clean step completed with stats: {stats}")
    logger.info(f"kept_count = {kept_count}")
    logger.info(f"drop_count = {drop_count}")
    
    return stats


def main() -> None:
    run_step_entrypoint(
        description="llm_data_pipeline.clean.run",
        run_func=run_clean,
        add_args_func=add_args,
        step_name="Clean",
    )
