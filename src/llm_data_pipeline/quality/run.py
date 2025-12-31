import argparse
import os

import ray.data as rd
from ray.data import ActorPoolStrategy

from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    resolve_io_paths,
    run_step_entrypoint,
    validate_input_path,
    validate_model_path,
    write_parquet,
)
from llm_data_pipeline.quality.model import LanguageFilter


def add_args(p: argparse.ArgumentParser):
    p.add_argument("--model-path", default="./models/lid.176.bin", help="Path to fasttext LID model")
    p.add_argument("--langs", default="zh,en", help="Comma separated languages to keep")
    p.add_argument("--threshold", type=float, default=0.4, help="LID confidence threshold")


def run_quality(config: PipelineConfig, **kwargs) -> dict:
    """Quality filtering step"""
    logger = PipelineLogger.get()

    # Resolve paths
    input_path, output_dir = resolve_io_paths(config, "quality", "deduped")

    # Validate input path
    validate_input_path(input_path, "quality")

    # Model path
    model_path = getattr(config, "model_path", kwargs.get("model_path", "./models/lid.176.bin"))
    validate_model_path(model_path, "quality")

    ds = rd.read_parquet(str(input_path.absolute()))

    limit = config.limit
    if limit > 0:
        logger.info(f"DEBUG: Limiting quality input to {limit} records.")
        ds = ds.limit(limit)
    logger.info(f"Reading {ds.count()} docs from {input_path}")

    langs_str = kwargs.get("langs", "zh,en")
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

    ds_scored = ds.map(QualityMapper(), compute=ActorPoolStrategy(min_size=1, max_size=os.cpu_count() or 4))

    # Filter
    ds_kept = ds_scored.filter(lambda r: r["quality_keep"])

    out_dir = output_dir / "quality_parquet"
    write_parquet(ds_kept, out_dir, logger)

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
