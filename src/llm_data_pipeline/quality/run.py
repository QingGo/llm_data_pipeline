import argparse
import logging
import os
from pathlib import Path

import ray
import ray.data as rd

from llm_data_pipeline.core import (
    PipelineConfig,
    resolve_io_paths,
    run_step_entrypoint,
)
from llm_data_pipeline.quality.model import LanguageFilter

logger = logging.getLogger(__name__)


def add_args(p: argparse.ArgumentParser):
    p.add_argument("--input", default=None, help="Input path")
    p.add_argument("--model-path", default="./models/lid.176.bin", help="Path to fasttext LID model")
    p.add_argument("--langs", default="zh,en", help="Comma separated languages to keep")
    p.add_argument("--threshold", type=float, default=0.4, help="LID confidence threshold")


def run_quality(config: PipelineConfig, **kwargs) -> dict:
    """Quality filtering step"""
    manual_input = kwargs.get("input")
    if manual_input:
        input_path = Path(manual_input)
        _, output_dir = resolve_io_paths(config, "quality")
    else:
        input_path, output_dir = resolve_io_paths(config, "quality", "deduped")

    # Model path
    model_path = getattr(config, "model_path", kwargs.get("model_path", "./models/lid.176.bin"))

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist for quality filter.")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}. Please download it first.")

    ds = rd.read_parquet(str(input_path.absolute()))

    limit = config.limit
    if limit > 0:
        logger.info(f"DEBUG: Limiting quality input to {limit} records.")
        ds = ds.limit(limit)
    logger.info(f"Reading {ds.count()} docs from {input_path}")

    langs_str = getattr(config, "langs", kwargs.get("langs", "zh,en"))
    langs = langs_str.split(",")
    threshold = getattr(config, "threshold", kwargs.get("threshold", 0.4))

    class QualityMapper:
        def __init__(self):
            self.filter = LanguageFilter(model_path=model_path, allowed_langs=langs, threshold=threshold)

        def __call__(self, row):
            text = row.get("text", "")
            keep, label, score = self.filter.keep(text)
            row["quality_keep"] = keep
            row["lang"] = label
            row["lang_score"] = score
            return row

    ds_scored = ds.map(QualityMapper, compute=ray.data.ActorPoolStrategy(min_size=1, max_size=os.cpu_count() or 4))  # pyright: ignore[reportArgumentType]

    # Filter
    ds_kept = ds_scored.filter(lambda r: r["quality_keep"])

    out_dir = output_dir / "quality_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing kept docs to {out_dir}...")
    ds_kept.write_parquet(str(out_dir))

    kept_count = ds_kept.count()
    orig_count = ds.count()
    logger.info(f"Done. Kept {kept_count} / {orig_count} docs.")

    return {"input_docs": orig_count, "kept_docs": kept_count, "output_path": str(out_dir)}


def main():
    run_step_entrypoint(
        description="llm_data_pipeline.quality.run",
        run_func=run_quality,
        add_args_func=add_args,
        step_name="Quality",
    )
