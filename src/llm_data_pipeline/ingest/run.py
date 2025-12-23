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


def main() -> None:
    """入口函数：初始化 Ray，构建数据集，派发解析并写出结果"""
    args = parse_args()
    venv_path = os.environ.get("VIRTUAL_ENV")
    ray.init(
        address=args.ray_address, object_store_memory=2 * 1024**3, runtime_env={"python": f"{venv_path}/bin/python"}
    )

    data_dir = Path(args.data_dir)
    files = discover_files(data_dir, args.pattern)
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    cfg = IngestConfig(
        min_text_chars=args.min_text_chars,
        max_text_chars=args.max_text_chars,
        max_docs_per_file=args.max_docs_per_file,
    )

    # 注意：这里不 read_binary_files（避免把大文件 bytes 搬进 object store）
    # 直接把“路径列表”做成 dataset，再在 worker 上打开文件解析。
    ds_files = rd.from_items([{"path": str(p.absolute())} for p in files])

    compute = rd.ActorPoolStrategy(size=args.taskpool_size) if args.taskpool_size else None

    ds_docs = ds_files.flat_map(
        lambda r: extract_wet_gz_file(Path(r["path"]), cfg),
        compute=compute,
        num_cpus=args.num_cpus,
    )
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    ds_docs.write_parquet(str(output_path.absolute()))

    print("files =", len(files))
    print("docs_count =", ds_docs.count())


if __name__ == "__main__":
    main()
