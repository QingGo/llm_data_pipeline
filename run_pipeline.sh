#!/bin/bash
set -e

# 1. Ingest (Output: outputs/dev/ingest_parquet)
echo "=== Step 1: Ingest ==="
uv run src/llm_data_pipeline/ingest/run.py --max-files 2 --output outputs/dev/ingest_parquet

# 2. Filter (Output: outputs/dev/cleaned_parquet)
echo "=== Step 2: Filter (Clean) ==="
uv run src/llm_data_pipeline/clean/run.py --input outputs/dev/ingest_parquet --output-dir outputs/dev


# 3. Dedup (MinHash -> Clustering)
echo "=== Step 3.1: Dedup (MinHash) ==="
uv run src/llm_data_pipeline/dedup/run.py --input outputs/dev/cleaned_parquet --output-dir outputs/dev

echo "=== Step 3.2: Dedup (Clustering) ==="
uv run src/llm_data_pipeline/dedup/run.py --input outputs/dev/minhash_parquet --output-dir outputs/dev --rows-per-band 4

# 3.5 Quality Filter (LID) (NEW)
echo "=== Step 3.5: Quality Filter (LID) ==="
chmod +x download_models.sh
./download_models.sh
uv run src/llm_data_pipeline/quality/run.py --input outputs/dev/deduped_parquet --output-dir outputs/dev --model-path models/lid.176.bin

# 4. Tokenizer (Train spm)
echo "=== Step 4: Train Tokenizer ==="
# Determine which data to use. Quality filtered data is best.
uv run src/llm_data_pipeline/tokenizer/train.py --parquet_dir outputs/dev/quality_parquet --work_dir outputs/dev/tokenizers/working --model_prefix outputs/dev/tokenizers/my_spm --vocab_size 32000

# 5. Tokenize & Pack
echo "=== Step 5: Tokenize & Pack ==="
# Uses quality filtered data
uv run src/llm_data_pipeline/tokenizer/run.py \
  --spm_model outputs/dev/tokenizers/my_spm.model \
  --input_dir outputs/dev/quality_parquet \
  --output_dir outputs/dev/token_packing_parquet \
  --seq_len 2048

# 6. Bin (Export)
echo "=== Step 6: Bin (Export) ==="
uv run src/llm_data_pipeline/export/run.py --input-dir outputs/dev/token_packing_parquet --output-file outputs/dev/final.bin --dtype uint16

echo "Pipeline finished successfully!"
