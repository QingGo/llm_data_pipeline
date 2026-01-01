"""WET.gz 数据抽取运行入口

发现输入文件、分发到 Ray 任务进行解析，并输出为 Parquet。
"""

import argparse
import time
from pathlib import Path

import ray.data as rd
from ray.data import ActorPoolStrategy

from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    get_directory_stats,
    resolve_io_paths,
    run_step_entrypoint,
    write_parquet,
)
from llm_data_pipeline.ingest.step import IngestConfig, extract_wet_gz_file


def discover_files(data_dir: Path, pattern: str) -> list[Path]:
    """按模式枚举数据目录下的文件，过滤隐藏文件"""
    logger = PipelineLogger.get()
    if not data_dir.exists():
        logger.warning(f"Data dir {data_dir} does not exist.")
        return []

    files = sorted(data_dir.glob(pattern))
    return list(filter(lambda p: not p.name.startswith("."), files))


def add_args(p: argparse.ArgumentParser) -> None:
    """添加 Ingest 特有参数"""
    p.add_argument("--data-dir", default="./data/commoncrawl/")
    p.add_argument("--pattern", default="**/*.wet.gz")
    p.add_argument("--taskpool-size", type=int, default=0, help="0=auto")
    p.add_argument("--num-cpus", type=float, default=1.0, help="cpus per task")
    p.add_argument("--min-text-chars", type=int, default=200)
    p.add_argument("--max-text-chars", type=int, default=200_000)
    p.add_argument("--max-docs-per-file", type=int, default=0)


def run_ingest(config: PipelineConfig, **kwargs) -> dict:
    """Pipeline entry point"""
    logger = PipelineLogger.get()

    total_start = time.time()

    # map args
    data_dir = Path(kwargs.get("data_dir", "./data/commoncrawl/"))
    pattern = kwargs.get("pattern", "**/*.wet.gz")

    # Path resolution
    _, output_base = resolve_io_paths(config, "ingest")
    output_path = output_base / "ingest_parquet"

    max_files = config.max_files
    limit = config.limit

    # Ingest specific args
    taskpool_size = kwargs.get("taskpool_size", 0)
    num_cpus = kwargs.get("num_cpus", 1.0)

    files = discover_files(data_dir, pattern)
    if max_files and max_files > 0:
        files = files[:max_files]

    # Calculate input stats
    input_file_count = len(files)
    input_total_size = sum(f.stat().st_size for f in files)

    cfg = IngestConfig(
        min_text_chars=kwargs.get("min_text_chars", 200),
        max_text_chars=kwargs.get("max_text_chars", 200_000),
        max_docs_per_file=kwargs.get("max_docs_per_file", 0),
    )

    if not files:
        logger.warning(f"No files found in {data_dir} with pattern {pattern}")
        total_duration = time.time() - total_start
        return {
            "step_name": "ingest",
            "input_path": str(data_dir),
            "input_file_count": input_file_count,
            "input_total_size": input_total_size,
            "input_count": 0,
            "output_path": str(output_path),
            "output_file_count": 0,
            "output_total_size": 0,
            "output_count": 0,
            "files_processed": 0,
            "docs_ingested": 0,
            "duration_seconds": total_duration,
            "duration_human": time.strftime("%H:%M:%S", time.gmtime(total_duration)),
            "process_duration_seconds": 0,
        }

    process_start = time.time()
    # dataset from items
    ds_files = rd.from_items([{"path": str(p.absolute())} for p in files])

    compute = ActorPoolStrategy(size=taskpool_size) if taskpool_size else None

    ds_docs = ds_files.flat_map(
        lambda r: extract_wet_gz_file(Path(r["path"]), cfg),
        compute=compute,
        num_cpus=num_cpus,
    )

    if limit > 0:
        logger.info(f"DEBUG: Limiting ingest to {limit} records.")
        ds_docs = ds_docs.limit(limit)

    write_parquet(ds_docs, output_path, logger)

    doc_count = ds_docs.count()
    process_end = time.time()
    total_end = time.time()

    # Get output stats
    output_file_count, output_total_size = get_directory_stats(output_path)

    # Prepare comprehensive stats in the same format as other steps
    stats = {
        "step_name": "ingest",
        "input_path": str(data_dir),
        "input_file_count": input_file_count,
        "input_total_size": input_total_size,
        "input_count": 0,  # Ingest stage creates new records, doesn't process existing ones
        "output_path": str(output_path),
        "output_file_count": output_file_count,
        "output_total_size": output_total_size,
        "output_count": doc_count,
        "files_processed": input_file_count,  # Keep for backward compatibility
        "docs_ingested": doc_count,  # Keep for backward compatibility
        "duration_seconds": total_end - total_start,
        "duration_human": time.strftime("%H:%M:%S", time.gmtime(total_end - total_start)),
        "process_duration_seconds": process_end - process_start,
    }

    return stats


def main() -> None:
    run_step_entrypoint(
        description="llm_data_pipeline.ingest.run (CommonCrawl WET.gz)",
        run_func=run_ingest,
        add_args_func=add_args,
        step_name="Ingest",
    )


if __name__ == "__main__":
    main()
