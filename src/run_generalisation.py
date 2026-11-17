"""M7: how well do the models work on designs they have never seen? (PRD experiment 6 and section 8.2)

Usage:
    python -m src.run_generalisation --data-root <dataset folder>
    python -m src.run_generalisation --plan configs/m7_generalisation.yaml --seeds 0 --max-epochs 2

For every variant config in the plan and every seed, three settings:
  random   - recipes split at random; every design is seen in training (as in M6)
  unseen   - by-design k-fold: each fold's model never sees that fold's designs; every design is held out once
  few_shot - the "unseen" model, fine-tuned on a few recipes of ONE held-out design, tested on its other recipes
"unseen" and "few_shot" are scored on exactly the same recipes (all except the shots), so they compare fairly.
Predictions from the 4 folds are pooled, so each "unseen"/"few_shot" score covers all designs.

Writes to --out (default results/generalisation): all_runs.csv (one row per variant, setting and seed),
per_design.csv (tau for each design), results_table.md / .csv (mean ± std over seeds) and generalisation_bar.png.
"""

import argparse
import copy
import csv
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data import (LABEL_COLUMNS, NUM_RECIPES, TargetScaler, build_datasets, load_design_graph,
                      load_labels, make_loader, make_split, target_base)
from src.evaluate import design_average_reference, metrics_from_real, predict, to_real
from src.models import build_model
from src.train import compute_loss, run
from src.utils import plot_experiment_bars, set_seed

METRICS = ["tau_and", "tau_delay", "mape_and", "mape_delay", "top10_and", "top10_delay"]
SETTINGS = ["random", "unseen", "few_shot"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generalisation experiments (random / unseen designs / few-shot).")
    p.add_argument("--plan", default="configs/m7_generalisation.yaml")
    p.add_argument("--variants", nargs="+", help="override the plan's variant configs")
    p.add_argument("--seeds", nargs="+", type=int, help="override the plan's seeds")
    p.add_argument("--data-root", help="override data.root in every config")
    p.add_argument("--max-epochs", type=int, help="override training.max_epochs (quick tests)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="results/generalisation")
    return p.parse_args(argv)


def load(path, args, seed):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["seed"] = seed
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.max_epochs:
        cfg["training"]["max_epochs"] = args.max_epochs
    return cfg


def check_folds(folds, designs):
    """Every design must be held out in exactly one fold."""
    held_out = [d for fold in folds for d in fold]
    if sorted(held_out) != sorted(designs):
        raise ValueError(f"folds {folds} must contain each of the designs {designs} exactly once")


def fold_config(cfg, k, fold):
    """The by-design version of a config: fold k's designs are held out for testing."""
    fcfg = copy.deepcopy(cfg)
    fcfg["experiment_name"] = f"{cfg['experiment_name']}_fold{k}"
    fcfg["data"].update(split="by_design", test_designs=list(fold),
                        split_file=str(Path(cfg["data"]["split_file"]).with_name(f"m7_fold{k}.json")))
    return fcfg


def few_shot_ids(seed, shots):
    """The recipe IDs used for fine-tuning: the same fixed random choice for every design, variant and seed."""
    return sorted(random.Random(seed).sample(range(NUM_RECIPES), shots))


# ---------------------------------------------------------------------------
# Predictions: (designs, syn_ids, true [N, T], pred [N, T]) in real units
# ---------------------------------------------------------------------------

def read_predictions(run_dir, target_names):
    """Test predictions that src.train.run saved in test_predictions.csv."""
    with open(Path(run_dir) / "test_predictions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return ([r["design"] for r in rows], [int(r["syn_id"]) for r in rows],
            np.array([[float(r[f"true_{t}"]) for t in target_names] for r in rows]),
            np.array([[float(r[f"pred_{t}"]) for t in target_names] for r in rows]))


def keep(preds, mask):
    designs, syn_ids, true, pred = preds
    mask = np.asarray(mask)
    return ([d for d, m in zip(designs, mask) if m], [s for s, m in zip(syn_ids, mask) if m], true[mask], pred[mask])


def concat(parts):
    return ([d for p in parts for d in p[0]], [s for p in parts for s in p[1]],
            np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]))


def fine_tune(model, dataset, train_cfg, ft, device, seed):
    """Copy the model and train the copy on a few samples for a fixed number of steps (no early stopping)."""
    set_seed(seed)
    tuned = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(tuned.parameters(), lr=ft["lr"])
    loader = make_loader(dataset, batch_size=len(dataset), shuffle=True, seed=seed)
    tuned.train()
    for _ in range(ft["epochs"]):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            compute_loss(tuned(batch), batch, train_cfg).backward()
            optimizer.step()
    return tuned


def few_shot_predictions(cfg, run_dir, ft, device):
    """For each held-out design of a fold: fine-tune the fold's best model on the shots, predict the rest."""
    data_cfg, train_cfg = cfg["data"], cfg["training"]
    names, mode = data_cfg["targets"], data_cfg["target_mode"]
    datasets, _, _ = build_datasets(data_cfg, data_cfg["split_seed"])
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True)
    model = build_model(cfg["model"], len(names)).to(device)
    model.load_state_dict(ckpt["model"])
    scaler = TargetScaler(torch.tensor(ckpt["scaler"]["mean"]), torch.tensor(ckpt["scaler"]["std"]))

    shots = set(few_shot_ids(data_cfg["split_seed"], ft["shots"]))
    test = datasets["test"]
    parts = []
    for design in data_cfg["test_designs"]:
        shot_idx = [i for i, (d, s) in enumerate(test.samples) if d == design and s in shots]
        rest_idx = [i for i, (d, s) in enumerate(test.samples) if d == design and s not in shots]
        tuned = fine_tune(model, test.subset(shot_idx), train_cfg, ft, device, cfg["seed"])
        out = predict(tuned, make_loader(test.subset(rest_idx), train_cfg["batch_size"]), device)
        parts.append((out["design"], out["syn_id"],
                      to_real(out["y"], out["base"], scaler, mode).numpy(),
                      to_real(out["pred"], out["base"], scaler, mode).numpy()))
    return concat(parts)


# ---------------------------------------------------------------------------
# Reference predictors (ignore the recipe; MAPE only, since a constant guess cannot rank)
# ---------------------------------------------------------------------------

def reference_rows(base_cfg, folds, ft):
    data_cfg = base_cfg["data"]
    names = data_cfg["targets"]
    labels = load_labels(data_cfg["root"])
    bases = {d: target_base(load_design_graph(data_cfg["root"], d)[1], names).numpy() for d in data_cfg["designs"]}
    cols = [LABEL_COLUMNS[t] for t in names]
    shots = few_shot_ids(data_cfg["split_seed"], ft["shots"])
    rest = [s for s in range(NUM_RECIPES) if s not in set(shots)]

    def score(guess, true, designs):
        m = metrics_from_real(np.array(guess), np.array(true), designs, names)
        return {k: m[k] for k in METRICS if k.startswith("mape")}

    refs = {}
    split = make_split({**data_cfg, "split": "random"}, data_cfg["split_seed"])
    refs["random"] = ("design average of its training recipes",
                      design_average_reference(data_cfg, split, labels))

    # Unseen: average ratio (final / initial) over the training designs, times the held-out design's initial value
    guess, true, designs = [], [], []
    for fold in folds:
        train = [d for d in data_cfg["designs"] if d not in fold]
        ratio = np.mean([labels[d][:, cols].numpy() / bases[d] for d in train], axis=(0, 1))
        for d in fold:
            for s in rest:
                guess.append(ratio * bases[d]); true.append(labels[d][s, cols].numpy()); designs.append(d)
    refs["unseen"] = ("average final/initial ratio of the training designs", score(guess, true, designs))

    # Few-shot: average of the design's own shot recipes
    guess, true, designs = [], [], []
    for d in data_cfg["designs"]:
        mean = labels[d][shots][:, cols].numpy().mean(axis=0)
        for s in rest:
            guess.append(mean); true.append(labels[d][s, cols].numpy()); designs.append(d)
    refs["few_shot"] = (f"average of the design's {ft['shots']} shot recipes", score(guess, true, designs))
    return refs


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def summarise(rows):
    """{(variant, setting): {metric: (mean, std, n)}} over seeds."""
    summary = {}
    for key in dict.fromkeys((r["variant"], r["setting"]) for r in rows):
        runs = [r for r in rows if (r["variant"], r["setting"]) == key]
        summary[key] = {m: (float(np.mean([r[m] for r in runs])), float(np.std([r[m] for r in runs])), len(runs))
                        for m in METRICS}
    return summary


def table_markdown(summary, refs):
    lines = ["| Variant | Setting | " + " | ".join(METRICS) + " | seeds |", "|" + "---|" * (len(METRICS) + 3)]
    for setting in SETTINGS:
        for (variant, s), stats in summary.items():
            if s == setting:
                cells = [f"{stats[m][0]:.3f} ± {stats[m][1]:.3f}" for m in METRICS]
                lines.append(f"| {variant} | {setting} | " + " | ".join(cells) + f" | {stats[METRICS[0]][2]} |")
        if setting in refs:
            name, values = refs[setting]
            cells = [f"{values[m]:.3f}" if m in values else "—" for m in METRICS]
            lines.append(f"| reference: {name} | {setting} | " + " | ".join(cells) + " | — |")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    plan = yaml.safe_load(Path(args.plan).read_text())
    variants, seeds = args.variants or plan["variants"], args.seeds or plan["seeds"]
    folds, ft = plan["folds"], plan["few_shot"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in variants:
        check_folds(folds, load(path, args, 0)["data"]["designs"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = ["variant", "setting", "seed"] + METRICS
    with open(out / "all_runs.csv", "w", newline="") as f:
        csv.writer(f).writerow(columns)
    rows, per_design, failures = [], [], []

    def record(variant, setting, seed, preds, names):
        m = metrics_from_real(preds[3], preds[2], preds[0], names)
        row = {"variant": variant, "setting": setting, "seed": seed, **{k: m[k] for k in METRICS}}
        rows.append(row)
        with open(out / "all_runs.csv", "a", newline="") as f:
            csv.writer(f).writerow([row[c] for c in columns])
        for d in sorted(set(preds[0])):
            per_design.append({"variant": variant, "setting": setting, "seed": seed, "design": d,
                               **{f"tau_{t}": m[f"tau_{t}_by_design"].get(d, float("nan")) for t in names}})
        print(f">>> {variant} | {setting} | seed {seed}: " + ", ".join(f"{k} {row[k]:.3f}" for k in METRICS))

    for path in variants:
        for seed in seeds:
            cfg = load(path, args, seed)
            name, names = cfg["experiment_name"], cfg["data"]["targets"]
            print(f"\n##### {name}  seed {seed}  random split #####")
            try:
                record(name, "random", seed, read_predictions(run(cfg, args.runs_dir), names), names)
            except Exception:
                traceback.print_exc()
                failures.append(f"{name} random seed {seed}")

            unseen, few = [], []
            try:
                shots = set(few_shot_ids(cfg["data"]["split_seed"], ft["shots"]))
                for k, fold in enumerate(folds):
                    fcfg = fold_config(cfg, k, fold)
                    print(f"\n##### {name}  seed {seed}  fold {k}: unseen designs {fold} #####")
                    run_dir = run(fcfg, args.runs_dir)
                    preds = read_predictions(run_dir, names)
                    unseen.append(keep(preds, [s not in shots for s in preds[1]]))   # score on non-shot recipes
                    few.append(few_shot_predictions(fcfg, run_dir, ft, device))
                record(name, "unseen", seed, concat(unseen), names)
                record(name, "few_shot", seed, concat(few), names)
            except Exception:
                traceback.print_exc()
                failures.append(f"{name} unseen/few-shot seed {seed}")

    refs = reference_rows(load(variants[0], args, seeds[0]), folds, ft)
    summary = summarise(rows)
    md = table_markdown(summary, refs)
    (out / "results_table.md").write_text(md + "\n", encoding="utf-8")
    with open(out / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "setting"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")] + ["seeds"])
        for (variant, setting), stats in summary.items():
            w.writerow([variant, setting] + [v for m in METRICS for v in stats[m][:2]] + [stats[METRICS[0]][2]])
    with open(out / "per_design.csv", "w", newline="") as f:
        if per_design:
            w = csv.DictWriter(f, fieldnames=list(per_design[0]))
            w.writeheader()
            w.writerows(per_design)
    if summary:
        plot_experiment_bars({f"{v} | {s}": stats for (v, s), stats in summary.items()},
                             ["tau_and", "tau_delay"], out / "generalisation_bar.png")

    print("\n===== RESULTS TABLE (mean ± std over seeds) =====")
    print(md)
    print("\n===== all_runs.csv =====")
    print((out / "all_runs.csv").read_text())
    print("===== per_design.csv =====")
    print((out / "per_design.csv").read_text())
    if failures:
        print("FAILED RUNS:", ", ".join(failures))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
