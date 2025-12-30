#!/bin/bash
chmod +x src/llm_data_pipeline/pipeline.py
uv run src/llm_data_pipeline/pipeline.py --steps ingest --max-files 2 > debug.log 2>&1
