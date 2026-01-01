import argparse
import json
from pathlib import Path

from llm_data_pipeline.clean.run import run_clean
from llm_data_pipeline.core import (
    PipelineConfig,
    init_ray,
    run_step,
    setup_logging,
    silence_ray_loggers,
)
from llm_data_pipeline.dedup.run_clustering import run_clustering
from llm_data_pipeline.dedup.run_minhash import run_minhash
from llm_data_pipeline.export.run import run_export

# Import steps (will refactor these modules next)
from llm_data_pipeline.ingest.run import run_ingest
from llm_data_pipeline.pii.run import run_pii
from llm_data_pipeline.quality.run import run_quality
from llm_data_pipeline.tokenizer.run import run_tokenize
from llm_data_pipeline.tokenizer.train import run_train_tokenizer


def main():
    p = argparse.ArgumentParser("LLM Data Pipeline Orchestrator")
    p.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma separated steps or 'all'. Available: ingest, clean, quality, pii, minhash, clustering, "
            "train_tokenizer, tokenize, export"
        ),
    )
    p.add_argument("--resume-from", default=None, help="Resume from this step (inclusive)")
    p.add_argument("--output-base", default="outputs/dev", help="Base output directory")
    p.add_argument("--ray-address", default="auto", help="Ray cluster address or 'local'")
    p.add_argument("--max-files", type=int, default=0, help="Debug: max files to ingest")
    p.add_argument("--limit", type=int, default=0, help="Debug: limit total records per step (0=no limit)")
    p.add_argument("--batch-size", type=int, default=4096, help="Batch size for Ray Data operations")
    p.add_argument("--concurrency", type=int, default=None, help="Concurrency for Ray Data operations (None=auto)")

    # Step specific consolidated args (can be expanded)
    p.add_argument("--langs", default="zh,en", help="Languages for quality filter")

    # PII specific args
    p.add_argument("--enable-ner", action="store_true", help="Enable PERSON NER stage (default: disabled)")
    p.add_argument("--text-col", default="text", help="Text column name (default: text)")
    p.add_argument("--lang-col", default="", help="Optional language column for PII")
    p.add_argument("--keep-stats", action="store_true", help="Keep pii_has_* columns in output")

    args = p.parse_args()

    out_dir = Path(args.output_base)
    logger = setup_logging(out_dir)

    # Init Ray via Core
    init_ray(args.ray_address, logger)

    # Build Config
    config = PipelineConfig(
        output_base=out_dir,
        ray_address=args.ray_address,
        max_files=args.max_files,
        limit=args.limit,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        resume_from=args.resume_from,
    )

    all_steps = [
        ("ingest", run_ingest),
        ("clean", run_clean),
        ("quality", run_quality),
        ("pii", run_pii),
        ("minhash", run_minhash),
        ("clustering", run_clustering),
        ("train_tokenizer", run_train_tokenizer),
        ("tokenize", run_tokenize),
        ("export", run_export),
    ]

    step_map = {name: func for name, func in all_steps}

    steps_to_run = []
    if args.steps == "all":
        steps_to_run = all_steps
    else:
        req_steps = args.steps.split(",")
        for r in req_steps:
            if r in step_map:
                steps_to_run.append((r, step_map[r]))
            else:
                logger.error(f"Unknown step: {r}")
                return

    # Handle resume-from
    if args.resume_from:
        start_idx = -1
        for i, (name, _) in enumerate(steps_to_run):
            if name == args.resume_from:
                start_idx = i
                break
        if start_idx == -1:
            logger.warning(f"Resume step {args.resume_from} not found in requested steps. Checking full list...")
            full_start_idx = -1
            for i, (name, _) in enumerate(all_steps):
                if name == args.resume_from:
                    full_start_idx = i
                    break
            if full_start_idx != -1:
                # If resume target is valid but not in current 'steps' scope,
                # we usually assume user wants to run from there onwards in the FULL list,
                # OR just relative to the filtered list.
                # Standard behavior: resume implies running the sequence starting at X.
                # If X is not in current list, it's ambiguous.
                # We'll take the tail of all_steps starting at X.
                steps_to_run = all_steps[full_start_idx:]
            else:
                logger.error(f"Resume step {args.resume_from} invalid.")
                return
        else:
            steps_to_run = steps_to_run[start_idx:]

    # Initialize stats dictionary to store all step statistics
    pipeline_stats = {}

    # Load existing stats if they exist
    stats_file = out_dir / "pipeline_stats.json"
    if stats_file.exists():
        try:
            with open(stats_file) as f:
                pipeline_stats = json.load(f)
            logger.info(f"Loaded existing stats from {stats_file}")
        except Exception as e:
            logger.warning(f"Failed to load existing stats: {e}")
            pipeline_stats = {}

    for name, func in steps_to_run:
        # Prepare extra args for specific steps if needed
        extra = {}
        if name == "quality":
            extra["langs"] = args.langs
        elif name == "pii":
            # PII specific args
            extra["enable_ner"] = args.enable_ner
            extra["text_col"] = args.text_col
            extra["lang_col"] = args.lang_col
            extra["keep_stats"] = args.keep_stats

        # Ensure we silence Ray again before each step
        silence_ray_loggers(logger)

        # Run step and capture stats
        step_stats = run_step(name, func, config, logger, extra_args=extra)

        # Store step stats
        pipeline_stats[name] = step_stats

        # Save stats after each step
        try:
            with open(stats_file, "w") as f:
                # Sort the stats by the order in all_steps to ensure consistent JSON order
                sorted_stats = {}
                for step_name, _ in all_steps:
                    if step_name in pipeline_stats:
                        sorted_stats[step_name] = pipeline_stats[step_name]
                json.dump(sorted_stats, f, indent=2, default=str)
            logger.info(f"Saved pipeline stats to {stats_file}")
        except Exception as e:
            logger.warning(f"Failed to save pipeline stats: {e}")

    # Sort the final stats by the order in all_steps to ensure consistent JSON order
    sorted_pipeline_stats = {}
    for step_name, _ in all_steps:
        if step_name in pipeline_stats:
            sorted_pipeline_stats[step_name] = pipeline_stats[step_name]

    logger.info("Pipeline finished successfully.")
    logger.info(f"Final pipeline stats: {json.dumps(sorted_pipeline_stats, indent=2, default=str)}")


if __name__ == "__main__":
    main()
