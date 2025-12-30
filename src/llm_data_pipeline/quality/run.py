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


def main():
    args = parse_args()
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    # Check model existence
    if not os.path.exists(args.model_path):
        raise RuntimeError(f"Model not found at {args.model_path}. Please download it first.")

    input_path = Path(args.input)
    ds = rd.read_parquet(str(input_path.absolute()))

    print(f"Reading {ds.count()} docs from {input_path}")

    langs = args.langs.split(",")
    # We load model in actor or use map_batches with class init to amortize load time?
    # Ray data Actors are best for this.

    class QualityMapper:
        def __init__(self):
            self.filter = LanguageFilter(model_path=args.model_path, allowed_langs=langs, threshold=args.threshold)

        def __call__(self, row):
            text = row.get("text", "")
            keep, label, score = self.filter.keep(text)
            row["quality_keep"] = keep
            row["lang"] = label
            row["lang_score"] = score
            return row

    # Ray Data 2.x: compute=ray.data.ActorPoolStrategy(size=...)
    # We use map with concurrency. Since model needs loading, let's just use simple map and rely on Ray's smarts
    # or explicitly use map_batches with setup if we want efficiency.
    # For now simple map with class-based callable (Ray will instantiate it).

    ds_scored = ds.map(QualityMapper, compute=ray.data.ActorPoolStrategy(min_size=1, max_size=os.cpu_count() or 4))

    # Filter
    ds_kept = ds_scored.filter(lambda r: r["quality_keep"])

    out_dir = Path(args.output_dir) / "quality_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing kept docs to {out_dir}...")
    ds_kept.write_parquet(str(out_dir))

    kept_count = ds_kept.count()
    print(f"Done. Kept {kept_count} / {ds.count()} docs.")


if __name__ == "__main__":
    main()
