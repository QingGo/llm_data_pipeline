import argparse
import os

import ray.data as rd
from ray.data import ActorPoolStrategy

from llm_data_pipeline.core import (
    PipelineConfig,
    PipelineLogger,
    run_step_entrypoint,
    step_wrapper,
    validate_model_path,
)
from llm_data_pipeline.quality.model import LanguageFilter


class QualityMapper:
    def __init__(self, model_path, allowed_langs, threshold):
        self.filter = LanguageFilter(model_path=model_path, allowed_langs=allowed_langs, threshold=threshold)

    def __call__(self, row):
        text = row.get("text", "")
        keep, label, score = self.filter.keep(text)
        row["quality_keep"] = keep
        row["lang"] = label
        row["lang_score"] = score
        return row


def add_args(p: argparse.ArgumentParser):
    p.add_argument("--model-path", default="./models/lid.176.bin", help="Path to fasttext LID model")
    p.add_argument("--langs", default="zh,en", help="Comma separated languages to keep")
    p.add_argument("--threshold", type=float, default=0.4, help="LID confidence threshold")


def _process_quality(ds: rd.Dataset, config: PipelineConfig, **kwargs) -> rd.Dataset:
    """Core quality filtering processing function"""
    logger = PipelineLogger.get()

    # Model path - fix: check if config.model_path is None before using it
    config_model_path = getattr(config, "model_path", None)
    model_path = (
        config_model_path if config_model_path is not None else kwargs.get("model_path", "./models/lid.176.bin")
    )
    # Get absolute model path
    model_path = validate_model_path(model_path, "quality")
    logger.info(f"Using model path: {model_path}")

    orig_count = ds.count()
    logger.info(f"Processing {orig_count} docs")

    # Handle empty dataset
    if orig_count == 0:
        logger.warning("Input dataset is empty. Skipping quality filtering.")
        return ds

    langs_str = kwargs.get("langs", "zh,en")
    langs = [lang.strip() for lang in langs_str.split(",") if lang.strip()]

    if not langs:
        raise ValueError("No languages specified. Please provide at least one language.")

    logger.info(f"Using languages: {langs}")

    # Threshold - fix: same issue, check if config.threshold is None before using it
    config_threshold = getattr(config, "threshold", None)
    threshold = config_threshold if config_threshold is not None else kwargs.get("threshold", 0.4)
    logger.info(f"Using language confidence threshold: {threshold}")

    logger.info("Initializing LanguageFilter with model...")
    logger.info("Starting language detection and scoring...")
    ds_scored = ds.map(
        QualityMapper,
        fn_constructor_args=(model_path, langs, threshold),
        compute=ActorPoolStrategy(min_size=1, max_size=os.cpu_count() or 4),
    )

    # Log language detection statistics
    logger.info("Language detection completed. Calculating statistics...")
    # Calculate language distribution
    lang_distribution = ds_scored.groupby("lang").count()
    logger.info(f"Language distribution: {lang_distribution}")

    # Calculate keep/filter statistics by language
    lang_keep_stats = ds_scored.groupby(["lang", "quality_keep"]).count()
    logger.info(f"Language keep statistics: {lang_keep_stats}")

    logger.info("Filtering documents based on language and confidence...")
    # Filter
    ds_kept = ds_scored.filter(lambda r: r["quality_keep"])

    kept_count = ds_kept.count()
    orig_count = ds.count()

    # Calculate statistics
    filtered_count = orig_count - kept_count
    keep_rate = kept_count / orig_count if orig_count > 0 else 0.0

    logger.info(f"Done. Kept {kept_count} / {orig_count} docs ({keep_rate:.2%}). Filtered out {filtered_count} docs.")

    # Log examples of kept and rejected docs
    if kept_count > 0:
        kept_sample = ds_kept.take(3)
        logger.info("Examples of kept docs:")
        for i, doc in enumerate(kept_sample):
            logger.info(f"  Kept {i + 1}: lang={doc['lang']}, score={doc['lang_score']:.4f}")

    if filtered_count > 0:
        rejected_sample = ds_scored.filter(lambda r: not r["quality_keep"]).take(3)
        logger.info("Examples of rejected docs:")
        for i, doc in enumerate(rejected_sample):
            logger.info(
                f"  Rejected {i + 1}: lang={doc['lang']}, score={doc['lang_score']:.4f}, "
                f"text_start={doc['text'][:50]}..."
            )

    return ds_kept


def run_quality(config: PipelineConfig, **kwargs) -> dict:
    """Quality filtering step"""
    stats = step_wrapper(
        step_name="quality", process_func=_process_quality, config=config, input_step_name="clean", **kwargs
    )

    # Add quality-specific stats
    orig_count = stats["input_count"]
    kept_count = stats["output_count"]
    filtered_count = orig_count - kept_count
    keep_rate = kept_count / orig_count if orig_count > 0 else 0.0

    stats.update({"kept_docs": kept_count, "filtered_docs": filtered_count, "keep_rate": keep_rate})

    return stats


def main():
    run_step_entrypoint(
        description="llm_data_pipeline.quality.run",
        run_func=run_quality,
        add_args_func=add_args,
        step_name="Quality",
    )
