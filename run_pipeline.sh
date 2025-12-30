#!/bin/bash
set -e

# Ensure models allow execution
chmod +x download_models.sh

# We could download models here, but the python pipeline checks for them in the quality step.
# If you want to ensure they exist before running:
# ./download_models.sh

echo "=== Starting Unified LLM Data Pipeline ==="
echo "Logs will be written to outputs/dev/pipeline.log"

# Forward all arguments to the python orchestrator
# Example usage:
# ./run_pipeline.sh --steps all
# ./run_pipeline.sh --steps ingest,clean --max-files 2
# ./run_pipeline.sh--steps ingest,clean,minhash --limit 1000 --output-base outputs/quick_test
uv run src/llm_data_pipeline/pipeline.py "$@"

