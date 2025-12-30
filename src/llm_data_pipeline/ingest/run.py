"""WET.gz 数据抽取运行入口

发现输入文件、分发到 Ray 任务进行解析，并输出为 Parquet。
"""

import argparse
import os
from pathlib import Path

import ray
import ray.data as rd

from llm_data_pipeline.ingest.step import IngestConfig, extract_wet_gz_file


def discover_files(data_dir: Path, pattern: str) -> list[Path]:
    """按模式枚举数据目录下的文件，过滤隐藏文件"""
    files = sorted(data_dir.glob(pattern))
    return list(filter(lambda p: not p.name.startswith("."), files))


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    p = argparse.ArgumentParser("llm_data_pipeline.ingest.run (CommonCrawl WET.gz)")
    p.add_argument("--data-dir", default="./data/commoncrawl/")
    p.add_argument("--pattern", default="**/*.wet.gz")
    p.add_argument("--output", default="./outputs/dev/ingest_parquet", help="e.g. outputs/dev/ingest_parquet")
    p.add_argument("--max-files", type=int, default=1, help="0=all (debug 可设 1~5)")
    p.add_argument("--ray-address", default=None, help='e.g. "auto" for cluster, default local')
    p.add_argument("--taskpool-size", type=int, default=0, help="0=auto")
    p.add_argument("--num-cpus", type=float, default=1.0, help="cpus per task")
    p.add_argument("--min-text-chars", type=int, default=200)
    p.add_argument("--max-text-chars", type=int, default=200_000)
    p.add_argument("--max-docs-per-file", type=int, default=0)
    return p.parse_args()


def run_ingest(args) -> dict:
    """Pipeline entry point"""
    # map args
    data_dir = Path(getattr(args, "data_dir", "./data/commoncrawl/"))
    pattern = getattr(args, "pattern", "**/*.wet.gz")

    # Use output_dir from pipeline if available, else default
    base_out = getattr(args, "output_dir", "./outputs/dev")
    output_path = Path(base_out) / "ingest_parquet"

    max_files = getattr(args, "max_files", 0)
    taskpool_size = getattr(args, "taskpool_size", 0)
    num_cpus = getattr(args, "num_cpus", 1.0)

    # ray.init is handled by pipeline

    files = discover_files(data_dir, pattern)
    if max_files and max_files > 0:
        files = files[:max_files]

    cfg = IngestConfig(
        min_text_chars=getattr(args, "min_text_chars", 200),
        max_text_chars=getattr(args, "max_text_chars", 200_000),
        max_docs_per_file=getattr(args, "max_docs_per_file", 0),
    )

    # dataset from items
    ds_files = rd.from_items([{"path": str(p.absolute())} for p in files])

    compute = rd.ActorPoolStrategy(size=taskpool_size) if taskpool_size else None

    ds_docs = ds_files.flat_map(
        lambda r: extract_wet_gz_file(Path(r["path"]), cfg),
        compute=compute,
        num_cpus=num_cpus,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Writing ingest result to {output_path}...")
    ds_docs.write_parquet(str(output_path.absolute()))

    doc_count = ds_docs.count()
    print("files =", len(files))
    print("docs_count =", doc_count)

    return {"files_processed": len(files), "docs_ingested": doc_count, "output_path": str(output_path)}


def main() -> None:
    """Cli wrapper"""
    args = parse_args()
    # Standalone mode: init ray
    venv_path = os.environ.get("VIRTUAL_ENV")
    ray.init(
        address=args.ray_address, object_store_memory=2 * 1024**3, runtime_env={"python": f"{venv_path}/bin/python"}
    )
    # Map cli args to what run_ingest expects
    # In standalone, args.output is explicit
    # run_ingest expects output_dir to build output_dir/ingest_parquet
    # We cheat a bit or adjust run_ingest.
    # Let's adjust run_ingest to respect 'output' if passed, else derive.

    # Actually, main() is rarely used now.
    # Just setting output_dir -> output.parent if possible?
    # run_ingest uses `output_dir` / `ingest_parquet`.
    # default args.output is outputs/dev/ingest_parquet.
    # so output_dir should be outputs/dev.

    args.output_dir = str(Path(args.output).parent)
    run_ingest(args)
