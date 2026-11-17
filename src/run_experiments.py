"""Run several experiment configs with several seeds and build the results table (PRD section 8.2).

Usage:
    python -m src.run_experiments --data-root <dataset folder>
    python -m src.run_experiments --configs configs/baseline.yaml configs/exp1_ratio.yaml --seeds 0 1

Writes to --out (default results/experiments): all_runs.csv (one row per run, appended as runs finish),
results_table.md / results_table.csv (mean ± std over seeds) and experiments_bar.png.
The split is fixed by data.split_seed, so only the model's random initialisation and shuffling change per seed.
"""

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np
import yaml

from src.data import load_labels, make_split
from src.evaluate import design_average_reference
from src.train import run
from src.utils import plot_experiment_bars

DEFAULT_CONFIGS = ["configs/baseline.yaml", "configs/exp1_ratio.yaml", "configs/exp2_gin.yaml",
                   "configs/exp2_sage.yaml", "configs/exp3_transformer.yaml", "configs/exp4_rankloss.yaml",
                   "configs/exp5_graph_only.yaml", "configs/exp5_recipe_only.yaml",
                   "configs/extra_select_tau.yaml"]
METRICS = ["tau_and", "tau_delay", "mape_and", "mape_delay", "top10_and", "top10_delay"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run experiments over several seeds.")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--data-root", help="override data.root in every config")
    p.add_argument("--max-epochs", type=int, help="override training.max_epochs (quick tests)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="results/experiments")
    return p.parse_args(argv)


def load(path, args, seed):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["seed"] = seed
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.max_epochs:
        cfg["training"]["max_epochs"] = args.max_epochs
    return cfg


def summarise(rows):
    """{experiment: {metric: (mean, std, n)}} over seeds, keeping the order experiments first appear in."""
    summary = {}
    for name in dict.fromkeys(r["experiment"] for r in rows):
        runs = [r for r in rows if r["experiment"] == name]
        summary[name] = {m: (float(np.mean([r[m] for r in runs])), float(np.std([r[m] for r in runs])), len(runs))
                         for m in METRICS}
    return summary


def table_markdown(summary, reference):
    head = "| Experiment | " + " | ".join(METRICS) + " | seeds |"
    lines = [head, "|" + "---|" * (len(METRICS) + 2)]
    for name, stats in summary.items():
        cells = [f"{stats[m][0]:.3f} ± {stats[m][1]:.3f}" for m in METRICS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {stats[METRICS[0]][2]} |")
    ref = [f"{reference[m]:.3f}" if m in reference else "—" for m in METRICS]
    lines.append("| design average (reference, ignores recipe) | " + " | ".join(ref) + " | — |")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    runs_csv = out / "all_runs.csv"
    columns = ["experiment", "seed", "best_epoch"] + METRICS + ["run_dir"]
    with open(runs_csv, "w", newline="") as f:
        csv.writer(f).writerow(columns)

    for path in args.configs:
        for seed in args.seeds:
            cfg = load(path, args, seed)
            print(f"\n##### {cfg['experiment_name']}  seed {seed} #####")
            try:
                run_dir = run(cfg, args.runs_dir)
            except Exception:
                traceback.print_exc()
                failures.append(f"{cfg['experiment_name']} seed {seed}")
                continue
            metrics = json.loads((run_dir / "test_metrics.json").read_text())
            row = {"experiment": cfg["experiment_name"], "seed": seed, "best_epoch": metrics["best_epoch"],
                   **{m: metrics[m] for m in METRICS}, "run_dir": str(run_dir)}
            rows.append(row)
            with open(runs_csv, "a", newline="") as f:
                csv.writer(f).writerow([row[c] for c in columns])

    # Reference row: same split as the experiments (taken from the first config)
    first = load(args.configs[0], args, args.seeds[0])
    data_cfg = first["data"]
    split = make_split(data_cfg, data_cfg.get("split_seed", first["seed"]))
    reference = design_average_reference(data_cfg, split, load_labels(data_cfg["root"]))

    summary = summarise(rows)
    md = table_markdown(summary, reference)
    (out / "results_table.md").write_text(md + "\n", encoding="utf-8")
    with open(out / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")] + ["seeds"])
        for name, stats in summary.items():
            w.writerow([name] + [v for m in METRICS for v in stats[m][:2]] + [stats[METRICS[0]][2]])
    if summary:
        plot_experiment_bars(summary, ["tau_and", "tau_delay"], out / "experiments_bar.png")

    print("\n===== RESULTS TABLE (test set, mean ± std over seeds) =====")
    print(md)
    print("\n===== all_runs.csv =====")
    print(runs_csv.read_text())
    if failures:
        print("FAILED RUNS:", ", ".join(failures))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
