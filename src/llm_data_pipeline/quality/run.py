import argparse
import os
from pathlib import Path

import ray
import ray.data as rd

from llm_data_pipeline.quality.model import LanguageFilter


def parse_args():
    p = argparse.ArgumentParser("llm_data_pipeline.quality.run")
    p.add_argument("--input", default="./outputs/dev/deduped_parquet", help="Input path")
    p.add_argument("--output-dir", default="./outputs/dev", help="Output parent dir")
    p.add_argument("--model-path", default="./models/lid.176.bin", help="Path to fasttext LID model")
    p.add_argument("--langs", default="zh,en", help="Comma separated languages to keep")
    p.add_argument("--threshold", type=float, default=0.4, help="LID confidence threshold")
    p.add_argument("--ray-address", default=None)
    return p.parse_args()


def run_quality(args) -> dict:
    """Quality filtering step"""
    base_out = getattr(args, "output_dir", "./outputs/dev")
    input_path = Path(getattr(args, "input", f"{base_out}/deduped_parquet"))
    output_dir = Path(base_out)

    # Model path - handle default relative to project root usually
    # or passed by pipeline. pipeline passes args including model_path if set.
    model_path = getattr(args, "model_path", "./models/lid.176.bin")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist for quality filter.")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}. Please download it first.")

    ds = rd.read_parquet(str(input_path.absolute()))

    limit = getattr(args, "limit", 0)
    if limit > 0:
        print(f"DEBUG: Limiting quality input to {limit} records.")
        ds = ds.limit(limit)
    print(f"Reading {ds.count()} docs from {input_path}")

    langs_str = getattr(args, "langs", "zh,en")
    langs = langs_str.split(",")
    threshold = getattr(args, "threshold", 0.4)

    # We load model in actor or use map_batches with class init to amortize load time?
    # Ray data Actors are best for this.
    class QualityMapper:
        def __init__(self):
            self.filter = LanguageFilter(model_path=model_path, allowed_langs=langs, threshold=threshold)

        def __call__(self, row):
            text = row.get("text", "")
            keep, label, score = self.filter.keep(text)
            row["quality_keep"] = keep
            row["lang"] = label
            row["lang_score"] = score
            return row

    ds_scored = ds.map(QualityMapper, compute=ray.data.ActorPoolStrategy(min_size=1, max_size=os.cpu_count() or 4))

    # Filter
    ds_kept = ds_scored.filter(lambda r: r["quality_keep"])

    out_dir = output_dir / "quality_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing kept docs to {out_dir}...")
    ds_kept.write_parquet(str(out_dir))

    kept_count = ds_kept.count()
    orig_count = ds.count()
    print(f"Done. Kept {kept_count} / {orig_count} docs.")

    return {"input_docs": orig_count, "kept_docs": kept_count, "output_path": str(out_dir)}


def main():
    args = parse_args()
    ray.init(address=args.ray_address, ignore_reinit_error=True)
    run_quality(args)
