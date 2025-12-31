import argparse
import dataclasses
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
    # Step-specific config storage
    _step_configs: dict[str, dict] = field(default_factory=dict, init=False)

    def get_step_config(self, step_name: str) -> dict:
        """Gets the configuration for a specific step."""
        return self._step_configs.get(step_name, {})

    def set_step_config(self, step_name: str, config: dict) -> None:
        """Sets the configuration for a specific step."""
        self._step_configs[step_name] = config

    def get(self, name: str, default: Any = None) -> Any:
        """Gets a configuration value, checking both the main config and step-specific configs."""
        return getattr(self, name, default)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        """Creates a PipelineConfig instance from argparse arguments."""
        # Extract main config fields
        main_fields = [f.name for f in dataclasses.fields(cls) if f.init]
        main_kwargs = {}

        for fld in main_fields:
            if hasattr(args, fld):
                value = getattr(args, fld)
                if fld == "output_base":
                    value = Path(value)
                elif fld in ["max_files", "limit", "batch_size"]:
                    value = int(value)
                main_kwargs[fld] = value

        return cls(**main_kwargs)


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
    Standard path resolution strategy for pipeline steps.

    This function implements the standard path resolution logic for pipeline steps:
    1. If an explicit input path is provided in the config, use that
    2. Otherwise, derive the input path from the previous step's output directory
    3. Return both the resolved input path and output directory

    The function follows these conventions:
    - Input path from previous step: base/<previous_step_name>_parquet
    - Special case for token_packing: base/token_packing_parquet
    - Special case for clean: base/cleaned_parquet
    - Output directory: base (each step will create its own subdirectory)

    Args:
        config: Pipeline configuration object
        step_name: Name of the current step
        input_step_name: Name of the previous step (if any)

    Returns:
        A tuple of (input_path, output_dir), both as Path objects
    """
    base = config.output_base
    output_dir = base
    logger = PipelineLogger.get()

    # Check if input is provided via config
    if hasattr(config, "input") and config.input:
        input_path = Path(config.input)
        logger.info(f"Step {step_name}: Using explicit input path from config: {input_path}")
        return input_path, output_dir

    # Input comes from previous step's parquet folder usually
    if input_step_name:
        # Convention: step X outputs to base/X_parquet
        # Special case: token_packing output is conventionally token_packing_parquet
        # Special case: clean output is conventionally cleaned_parquet
        if input_step_name == "token_packing":
            input_path = base / "token_packing_parquet"
        elif input_step_name == "clean":
            input_path = base / "cleaned_parquet"
        else:
            input_path = base / f"{input_step_name}_parquet"
        logger.info(f"Step {step_name}: Using derived input path from {input_step_name}: {input_path}")
    else:
        input_path = base  # Fallback when no previous step
        logger.info(f"Step {step_name}: Using fallback input path: {input_path}")

    return input_path, output_dir


def read_parquet(input_path: Path, config: PipelineConfig) -> rd.Dataset:
    """
    Reads parquet files from the input path and applies limit if specified.

    This function handles reading parquet files using Ray Data, with support for:
    1. Reading from a directory containing multiple parquet files
    2. Applying a record limit for debugging purposes
    3. Logging the read operation

    Args:
        input_path: Path to the directory containing parquet files
        config: Pipeline configuration object, which may include a limit

    Returns:
        A Ray Dataset containing the read records
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
    import shutil

    # Clear the output directory if it exists
    if output_path.exists():
        logger.info(f"Clearing output directory {output_path}...")
        shutil.rmtree(output_path)

    # Create the output directory
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to {output_path}...")
    ds.write_parquet(str(output_path.absolute()))


def run_step(
    step_name: str,
    func: Callable,
    config: PipelineConfig,
    logger: logging.Logger,
    extra_args: dict[str, Any] | None = None,
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

        # Add processing time to stats
        stats["duration_seconds"] = duration
        stats["duration_human"] = time.strftime("%H:%M:%S", time.gmtime(duration))

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

    This function provides a consistent entry point for all pipeline steps, handling:
    1. Argument parsing with common and step-specific arguments
    2. Logging setup with file and console output
    3. Ray initialization (if enabled)
    4. Running the step function with proper error handling
    5. Exit code management

    Args:
        description: Description of the step for the argument parser
        run_func: The main step function to execute, should accept PipelineConfig as first argument
        add_args_func: Optional function to add step-specific arguments to the parser
        step_name: Optional name for the step, defaults to first word of description
        use_ray: Whether to initialize Ray for this step
    """
    # Create argument parser with common arguments
    p = get_arg_parser(description)

    # Add step-specific arguments if provided
    if add_args_func:
        add_args_func(p)

    # Parse command line arguments
    args = p.parse_args()

    # Derive step name from description if not provided
    if not step_name:
        step_name = description.split(" ")[0]

    # Create PipelineConfig from parsed arguments
    config = PipelineConfig.from_args(args)

    # Setup logging for the step
    logger = setup_logging(config.output_base, step_name=step_name)

    # Initialize Ray if enabled
    if use_ray:
        init_ray(config.ray_address, logger)

    try:
        # Execute the step function with config and all parsed arguments
        # This allows the step function to access both structured config and raw arguments
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


def validate_model_path(model_path: str | Path, step_name: str) -> str:
    """
    Validates that the model path exists and returns the absolute path.
    """
    # Convert to Path object to handle relative paths
    model_path = Path(model_path)
    # Get absolute path
    abs_model_path = model_path.absolute()
    if not abs_model_path.exists():
        raise RuntimeError(f"Model not found at {abs_model_path} for {step_name} step. Please download it first.")
    return str(abs_model_path)
