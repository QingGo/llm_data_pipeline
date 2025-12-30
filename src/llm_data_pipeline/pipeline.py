import argparse
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import ray

# Import steps (will refactor these modules next)
# Using lazy imports inside functions or ensuring they are importable
# We will refactor run.py files to expose run_* functions.


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"

    # Remove existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("Pipeline")


def run_step(step_name: str, func, args, logger):
    logger.info(f"=== Starting Step: {step_name} ===")
    start_time = time.time()
    try:
        stats = func(args)
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
        help="Comma separated steps or 'all'. Available: ingest, clean, minhash, clustering, quality, train_tokenizer, tokenize, export",
    )
    p.add_argument("--resume-from", default=None, help="Resume from this step (inclusive)")
    p.add_argument("--output-base", default="outputs/dev", help="Base output directory")
    p.add_argument("--ray-address", default="auto", help="Ray cluster address or 'local'")
    # Add other common args or let steps parse their own from a config dict

    # We will pass a config object to steps, populated from CLI + Defaults
    p.add_argument("--max-files", type=int, default=0, help="Debug: max files to ingest")

    args = p.parse_args()

    out_dir = Path(args.output_base)
    logger = setup_logging(out_dir)

    logger.info(f"Initializing Ray (address={args.ray_address})...")
    # Redirect Ray logs to storage to keep terminal clean-ish?
    # Ray usually prints to stderr. We can't easily suppress it without setting env vars before init.
    # But user asked: "ray 框架的输出应该只输出到日志文件".
    # We can try to redirect stderr, but that might hide actual errors.
    # Better to configure Ray logging.

    ray.init(
        address=args.ray_address if args.ray_address != "local" else None, ignore_reinit_error=True, log_to_driver=True
    )  # log_to_driver=True ensures we capture it.

    # Define steps
    # We need to import the functions now. I will assume they exist for the plan.
    from llm_data_pipeline.clean.run import run_clean
    from llm_data_pipeline.dedup.minhash_step import run_minhash  # New file
    from llm_data_pipeline.dedup.run import run_clustering
    from llm_data_pipeline.export.run import run_export
    from llm_data_pipeline.ingest.run import run_ingest
    from llm_data_pipeline.quality.run import run_quality
    from llm_data_pipeline.tokenizer.run import run_tokenize
    from llm_data_pipeline.tokenizer.train import run_train_tokenizer

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
            # Maybe it wasn't in the explicit list, try to find in all steps
            logger.warning(f"Resume step {args.resume_from} not found in requested steps. Checking full list...")
            full_start_idx = -1
            for i, (name, _) in enumerate(all_steps):
                if name == args.resume_from:
                    full_start_idx = i
                    break
            if full_start_idx != -1:
                steps_to_run = all_steps[full_start_idx:]
            else:
                logger.error(f"Resume step {args.resume_from} invalid.")
                return
        else:
            steps_to_run = steps_to_run[start_idx:]

    context = {
        "output_base": args.output_base,
        "max_files": args.max_files,
        # Add other shared configs
    }

    for name, func in steps_to_run:
        # We pass a mixed args object or dict.
        # Refactoring run_* to accept a generic object or dict is best.
        # I'll create a Namespace-like object merging args and context

        step_args = argparse.Namespace(**vars(args))
        # Add derived paths
        step_args.output_dir = args.output_base
        # Specific step defaults can be handled inside step functions if missing

        run_step(name, func, step_args, logger)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
