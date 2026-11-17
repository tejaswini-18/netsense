"""Write a compact copy of the data the pipeline reads (e.g. for a Kaggle upload).

The original dataset is 63,000 zipped graph files, but the pipeline only needs:
  - each design's initial graph (identical in all 1500 files of a design), and
  - the 1500 recipes (identical for every design), and
  - synthesisStatistics.pickle (labels).
This writes them as graphs/<design>.pt (plain tensors), recipes.json and the labels pickle:
44 files of about 25 MB, under Kaggle's 1000-file upload limit and with no nested .zip files.

Usage:
    python -m src.make_subset --data-root OPENABC2_DATASET --out kaggle_data/openabc-mini
The source folder is only read, never modified.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import torch

from src.data import compact_graph_file, load_raw_graph, load_recipes


def make_subset(data_root, out, recipe_source="ss_pcm", designs=None):
    data_root, out = Path(data_root), Path(out)
    if out.resolve() == data_root.resolve():
        raise ValueError("output folder must differ from the dataset folder")
    with open(data_root / "synthesisStatistics.pickle", "rb") as f:
        designs = designs or sorted(pickle.load(f))

    (out / "graphs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_root / "synthesisStatistics.pickle", out / "synthesisStatistics.pickle")
    recipes = load_recipes(data_root, recipe_source)
    (out / "recipes.json").write_text(json.dumps(recipes.tolist()))
    for d in designs:
        torch.save(load_raw_graph(data_root, d), compact_graph_file(out, d))

    files = [p for p in out.rglob("*") if p.is_file()]
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"Wrote {len(files)} files ({len(designs)} designs) to {out}: {size_mb:.1f} MB")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", required=True, help="full OpenABC-D folder (read only)")
    p.add_argument("--out", required=True, help="new folder to create")
    p.add_argument("--recipe-source", default="ss_pcm")
    args = p.parse_args()
    make_subset(args.data_root, args.out, args.recipe_source)


if __name__ == "__main__":
    main()
