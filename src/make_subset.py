"""Copy the few dataset files the pipeline actually reads into a small folder (e.g. for a Kaggle upload).

The pipeline needs, per design, only recipe 0's graph file (the graph is the same in all 1500 files),
the 1500 small files of the recipe-source design (to read the recipes), and synthesisStatistics.pickle.
For all 42 designs that is about 25 MB instead of 18 GB.

Usage:
    python -m src.make_subset --data-root OPENABC2_DATASET --out kaggle_data/openabc-mini
The source folder is only read, never modified.
"""

import argparse
import pickle
import shutil
from pathlib import Path

from src.data import NUM_RECIPES, graph_file


def make_subset(data_root, out, recipe_source="ss_pcm", designs=None):
    data_root, out = Path(data_root), Path(out)
    if out.resolve() == data_root.resolve():
        raise ValueError("output folder must differ from the dataset folder")
    with open(data_root / "synthesisStatistics.pickle", "rb") as f:
        all_designs = sorted(pickle.load(f))
    designs = designs or all_designs

    (out / "processed").mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_root / "synthesisStatistics.pickle", out / "synthesisStatistics.pickle")
    files = {graph_file(data_root, d, 0) for d in designs}
    files |= {graph_file(data_root, recipe_source, i) for i in range(NUM_RECIPES)}
    for src in sorted(files):
        shutil.copy2(src, out / "processed" / src.name)
    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Copied {len(files) + 1} files ({len(designs)} designs) to {out}: {size_mb:.1f} MB")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", required=True, help="full OpenABC-D folder (read only)")
    p.add_argument("--out", required=True, help="new folder to create")
    p.add_argument("--recipe-source", default="ss_pcm")
    args = p.parse_args()
    make_subset(args.data_root, args.out, args.recipe_source)


if __name__ == "__main__":
    main()
