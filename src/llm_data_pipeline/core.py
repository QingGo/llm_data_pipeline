import argparse
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
import ray.data as rd


@dataclass
class PipelineConfig:
    """Shared configuration for pipeline steps."""

    output_base: Path
    ray_address: str | None = None
    max_files: int = 0
    limit: int = 0
    batch_size: int = 4096
    concurrency: int | None = None
    resume_from: str | None = None
    # Additional common config options
    model_path: str | None = None
    input: str | None = None
    threshold: float | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        """Creates a PipelineConfig instance from argparse arguments."""
        return cls(
            output_base=Path(getattr(args, "output_base", "outputs/dev")),
            ray_address=getattr(args, "ray_address", "auto"),
            max_files=int(getattr(args, "max_files", 0)),
            limit=int(getattr(args, "limit", 0)),
            batch_size=int(getattr(args, "batch_size", 4096)),
            concurrency=getattr(args, "concurrency", None),
            resume_from=getattr(args, "resume_from", None),
            model_path=getattr(args, "model_path", None),
            input=getattr(args, "input", None),
            threshold=getattr(args, "threshold", None),
        )


class PipelineLogger:
    """Singleton logger for the pipeline."""

    _instance: logging.Logger | None = None

    @classmethod
    def get(cls) -> logging.Logger:
        """Gets the singleton logger instance."""
        if cls._instance is None:
            raise RuntimeError("Logger not initialized. Call setup_logging first.")
        return cls._instance

    @classmethod
    def set(cls, logger: logging.Logger) -> None:
        """Sets the singleton logger instance."""
        cls._instance = logger


def setup_logging(output_dir: Path, step_name: str = "Pipeline") -> logging.Logger:
    """Sets up logging to file and terminal."""
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # If running a specific step standalone, maybe use step name in log
    log_file = logs_dir / f"pipeline_{timestamp}.log"

    # Remove existing handlers to avoid duplicates if re-inited
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

    # Configure Step/Pipeline logger
    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.propagate = True  # Propagate to Root (File)

    logger.info(f"Logging to {log_file}")
    PipelineLogger.set(logger)
    return logger


def silence_ray_loggers(logger: logging.Logger):
    """
    Configure Ray loggers to write ONLY to the pipeline log file and NOT to the terminal.
    """
    # Find root file handler
    root_file_handler = None
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.FileHandler):
            root_file_handler = h
            break

    if not root_file_handler:
        return

    # 1. Disable Ray Data Progress Bars
    try:
        from ray.data import DataContext

        DataContext.get_current().enable_progress_bars = False
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not disable Ray Data progress bars: {e}")

    # 2. Configure loggers
    target_loggers = ["ray.data", "ray", "ray.serve"]
    for name in target_loggers:
        lg = logging.getLogger(name)
        lg.propagate = False
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        lg.addHandler(root_file_handler)


def init_ray(address: str | None, logger: logging.Logger):
    """Initialize Ray with proper logging configuration."""
    if ray.is_initialized():
        return

    logger.info(f"Initializing Ray (address={address})...")
    ray.init(
        address=address if address != "local" else None,
        ignore_reinit_error=True,
        log_to_driver=False,
        logging_level=logging.INFO,
        configure_logging=False,
    )
    # Apply silencing after init
    silence_ray_loggers(logger)


def get_arg_parser(description: str) -> argparse.ArgumentParser:
    """Returns a parser with common arguments."""
    p = argparse.ArgumentParser(description)
    p.add_argument("--output-base", default="outputs/dev", help="Base output directory")
    p.add_argument("--ray-address", default="auto", help="Ray cluster address or 'local'")
    p.add_argument("--limit", type=int, default=0, help="Debug: limit total records (0=no limit)")
    p.add_argument("--batch-size", type=int, default=4096, help="Batch size for Ray Data")
    p.add_argument("--concurrency", type=int, default=None, help="Concurrency (None=auto)")
    p.add_argument("--input", default=None, help="Input directory (default: <output_base>/<previous_step>_parquet)")
    return p


def resolve_io_paths(config: PipelineConfig, step_name: str, input_step_name: str | None = None) -> tuple[Path, Path]:
    """
    Standard path resolution strategy.
    Returns (input_path, output_dir).
    """
    base = config.output_base
    output_dir = base

    # Check if input is provided via config
    if hasattr(config, "input") and config.input:
        return Path(config.input), output_dir

    # Input comes from previous step's parquet folder usually
    if input_step_name:
        # Convention: step X outputs to base/X_parquet
        # Special case: token_packing output is conventionally token_packing_parquet
        if input_step_name == "token_packing":
            input_path = base / "token_packing_parquet"
        else:
            input_path = base / f"{input_step_name}_parquet"
    else:
        input_path = base  # Fallback

    return input_path, output_dir


def read_parquet(input_path: Path, config: PipelineConfig) -> rd.Dataset:
    """
    Reads parquet files from the input path and applies limit if specified.
    """
    logger = PipelineLogger.get()
    logger.info(f"Reading parquet from {input_path}")
    ds = rd.read_parquet(str(input_path.absolute()))

    if config.limit > 0:
        logger.info(f"DEBUG: Limiting input to {config.limit} records.")
        ds = ds.limit(config.limit)

    return ds


def write_parquet(ds: rd.Dataset, output_path: Path, logger: logging.Logger) -> None:
    """
    Writes a Ray Dataset to parquet files.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to {output_path}...")
    ds.write_parquet(str(output_path.absolute()))


def run_step(
    step_name: str,
    func: Callable,
    config: PipelineConfig,
    logger: logging.Logger,
    extra_args: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Runs a pipeline step with logging and error handling.
    """
    logger.info(f"=== Starting Step: {step_name} ===")
    start_time = time.time()
    try:
        stats = func(config, **(extra_args or {}))

        duration = time.time() - start_time
        logger.info(f"=== Finished Step: {step_name} in {time.strftime('%H:%M:%S', time.gmtime(duration))} ===")
        logger.info(f"Stats: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Step {step_name} failed: {e}", exc_info=True)
        raise


def run_step_entrypoint(
    description: str,
    run_func: Callable,
    add_args_func: Callable | None = None,
    step_name: str | None = None,
    use_ray: bool = True,
) -> None:
    """
    Standard entry point for pipeline steps.
    Handles arg parsing, logging setup, Ray init, and running the step.
    """
    p = get_arg_parser(description)
    if add_args_func:
        add_args_func(p)

    args = p.parse_args()

    # Deriving step name from description if not provided
    if not step_name:
        step_name = description.split(" ")[0]

    config = PipelineConfig.from_args(args)
    logger = setup_logging(config.output_base, step_name=step_name)

    if use_ray:
        init_ray(config.ray_address, logger)

    try:
        # Convert args to Config if the function expects it
        # We pass config as the first argument.
        # We also pass **vars(args) to allow access to extra arguments not in Config
        # This allows functions to signature match like (config: PipelineConfig, **kwargs)
        run_func(config, **vars(args))
    except Exception as e:
        logger.error(f"Step {step_name} failed: {e}", exc_info=True)
        sys.exit(1)


def validate_input_path(input_path: Path, step_name: str) -> None:
    """
    Validates that the input path exists.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist for {step_name} step.")


def validate_model_path(model_path: str | Path, step_name: str) -> None:
    """
    Validates that the model path exists.
    """
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path} for {step_name} step. Please download it first.")

