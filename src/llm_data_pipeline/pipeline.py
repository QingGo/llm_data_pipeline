import argparse
import time
from datetime import timedelta
from pathlib import Path

from llm_data_pipeline.clean.run import run_clean
from llm_data_pipeline.core import (
    PipelineConfig,
    init_ray,
    setup_logging,
    silence_ray_loggers,
)
from llm_data_pipeline.dedup.minhash_step import run_minhash
from llm_data_pipeline.dedup.run import run_clustering
from llm_data_pipeline.export.run import run_export

# Import steps (will refactor these modules next)
from llm_data_pipeline.ingest.run import run_ingest
from llm_data_pipeline.quality.run import run_quality
from llm_data_pipeline.tokenizer.run import run_tokenize
from llm_data_pipeline.tokenizer.train import run_train_tokenizer


def run_step(step_name: str, func, config: PipelineConfig, logger, extra_args: dict | None = None):
    logger.info(f"=== Starting Step: {step_name} ===")
    start_time = time.time()
    try:
        stats = func(config, **(extra_args or {}))

        duration = time.time() - start_time
        logger.info(f"=== Finished Step: {step_name} in {timedelta(seconds=duration)} ===")
        logger.info(f"Stats: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Step {step_name} failed: {e}", exc_info=True)
        raise


def main():
    p = argparse.ArgumentParser("LLM Data Pipeline Orchestrator")
    p.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma separated steps or 'all'. Available: ingest, clean, minhash, clustering, "
            "quality, train_tokenizer, tokenize, export"
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
        ("minhash", run_minhash),
        ("clustering", run_clustering),
        ("quality", run_quality),
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

    for name, func in steps_to_run:
        # Prepare extra args for specific steps if needed
        extra = {}
        if name == "quality":
            extra["langs"] = args.langs

        # Ensure we silence Ray again before each step
        silence_ray_loggers(logger)

        run_step(name, func, config, logger, extra_args=extra)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
