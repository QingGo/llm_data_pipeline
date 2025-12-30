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
    # Create global logs dir
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pipeline_{timestamp}.log"

    # Remove existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    class TerminalFormatter(logging.Formatter):
        """Minimal formatter for terminal output"""

        def format(self, record):
            return f"[{record.levelname}] {record.getMessage()}"

    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(TerminalFormatter())

    # Basic config sets up Root logger
    logging.basicConfig(level=logging.INFO, handlers=[file_handler])

    # Configure Pipeline logger to also print to console
    pipeline_logger = logging.getLogger("Pipeline")
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.addHandler(stream_handler)
    pipeline_logger.propagate = True  # Propagate to Root (File)

    pipeline_logger.info(f"Logging to {log_file}")
    return pipeline_logger


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
    p.add_argument("--limit", type=int, default=0, help="Debug: limit total records per step (0=no limit)")
    p.add_argument("--batch-size", type=int, default=4096, help="Batch size for Ray Data operations")
    p.add_argument("--concurrency", type=int, default=None, help="Concurrency for Ray Data operations (None=auto)")

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
        address=args.ray_address if args.ray_address != "local" else None,
        ignore_reinit_error=True,
        log_to_driver=False,
        logging_level=logging.INFO,
        configure_logging=False,  # We handle logging
    )

    # Force Ray logs to file only (do this AFTER init as Ray might re-add handlers)
    # We grab the file handler from the root logger (set by setup_logging)
    root_file_handler = None
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.FileHandler):
            root_file_handler = h
            break

    def silence_ray_loggers():
        """
        Configure Ray loggers to write ONLY to the pipeline log file and NOT to the terminal.
        """
        if not root_file_handler:
            return

        # 1. Disable Ray Data Progress Bars
        try:
            from ray.data import DataContext

            DataContext.get_current().enable_progress_bars = False
        except ImportError:
            pass
        except Exception as e:
            # If ray isn't fully set up yet, this might fail, but it's usually safe
            logger.warning(f"Could not disable Ray Data progress bars: {e}")

        # 2. Configure loggers
        # We focus on the main culprits identified by the user and docs
        # ray.data is the one producing "Running Dataset..." (if progress bars are on) and INFO logs
        target_loggers = ["ray.data", "ray", "ray.serve"]

        for name in target_loggers:
            lg = logging.getLogger(name)
            # Ensure it doesn't print to stdout/stderr (root)
            lg.propagate = False
            # Set level to INFO so we still capture useful info in the file (as requested by user)
            lg.setLevel(logging.INFO)

            # Clear existing handlers (like default StreamHandlers)
            lg.handlers.clear()

            # Add OUR file handler so it goes to the file
            lg.addHandler(root_file_handler)

    # Call it once after init
    silence_ray_loggers()

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
        "limit": args.limit,
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

        # Ensure we silence Ray again before each step as lazy loading might have reset it
        silence_ray_loggers()

        run_step(name, func, step_args, logger)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
