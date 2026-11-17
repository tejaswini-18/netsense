"""Train and evaluate one experiment.

Usage:
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/baseline.yaml --data-root /kaggle/input/openabc-subset --max-epochs 2

Writes runs/<experiment_name>_<timestamp>/ with: config.yaml, metrics.csv (one row per epoch),
best.pt (best checkpoint), test_metrics.json, test_predictions.csv, loss_curve.png, test_scatter.png.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import TargetScaler, build_datasets
from src.evaluate import compute_metrics, predict
from src.models import build_model, count_parameters
from src.utils import make_run_dir, plot_loss_curves, plot_pred_vs_actual, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a QoR prediction model.")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--data-root", help="override data.root (e.g. the Kaggle dataset path)")
    p.add_argument("--max-epochs", type=int, help="override training.max_epochs (e.g. for a quick test)")
    p.add_argument("--runs-dir", default="runs", help="where run folders are created")
    return p.parse_args(argv)


def load_config(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.max_epochs:
        cfg["training"]["max_epochs"] = args.max_epochs
    return cfg


def train_one_epoch(model, loader, optimizer, device):
    """One pass over the training data; returns the average training loss."""
    model.train()
    total, count = 0.0, 0
    for batch in tqdm(loader, desc="  train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(batch), batch.y)
        loss.backward()           # compute gradients
        optimizer.step()          # update the weights
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / count


def summary_line(prefix, metrics, target_names):
    parts = [f"loss {metrics['loss']:.4f}"]
    for t in target_names:
        parts.append(f"MAPE_{t} {metrics[f'mape_{t}']:.2f}%  tau_{t} {metrics[f'tau_{t}']:.3f}")
    return f"{prefix} " + " | ".join(parts)


def run(cfg, runs_dir="runs"):
    """Train with early stopping, then evaluate the best checkpoint on the test set. Returns the run folder."""
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, train_cfg = cfg["data"], cfg["training"]
    target_names, mode = data_cfg["targets"], data_cfg["target_mode"]

    run_dir = make_run_dir(cfg["experiment_name"], runs_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Run folder: {run_dir}  |  device: {device}")

    datasets, scaler, meta = build_datasets(data_cfg, cfg["seed"])
    print(f"Samples: {meta['split_sizes']}")
    # A seeded generator makes the shuffling order reproducible
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=train_cfg["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(cfg["seed"])),
        "val": DataLoader(datasets["val"], batch_size=train_cfg["batch_size"]),
        "test": DataLoader(datasets["test"], batch_size=train_cfg["batch_size"]),
    }

    model = build_model(cfg["model"], num_targets=len(target_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    print(f"Model parameters: {count_parameters(model):,}")

    history, best_val, best_epoch, waited = [], float("inf"), 0, 0
    columns = ["epoch", "train_loss", "val_loss"] + [f"val_{m}_{t}" for t in target_names
                                                     for m in ("mape", "tau", "top10")] + ["seconds"]
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(columns)

    for epoch in range(1, train_cfg["max_epochs"] + 1):
        start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device)
        val, _, _ = compute_metrics(predict(model, loaders["val"], device), scaler, mode, target_names)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"], "seconds": time.time() - start}
        for t in target_names:
            for m in ("mape", "tau", "top10"):
                row[f"val_{m}_{t}"] = val[f"{m}_{t}"]
        history.append(row)
        with open(run_dir / "metrics.csv", "a", newline="") as f:
            csv.writer(f).writerow([row[c] for c in columns])

        # Early stopping: keep the best model by validation loss, stop when it stops improving
        improved = val["loss"] < best_val
        if improved:
            best_val, best_epoch, waited = val["loss"], epoch, 0
            torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(), "config": cfg,
                        "epoch": epoch}, run_dir / "best.pt")
        else:
            waited += 1
        print(f"epoch {epoch:3d} train {train_loss:.4f} | " + summary_line("val", val, target_names)
              + f" | {row['seconds']:.0f}s" + ("  *best*" if improved else ""))
        if waited >= train_cfg["early_stopping_patience"]:
            print(f"Early stopping: no improvement for {waited} epochs.")
            break

    # Evaluate the best checkpoint (not the last epoch) on the test set, exactly once
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    scaler = TargetScaler(torch.tensor(ckpt["scaler"]["mean"]), torch.tensor(ckpt["scaler"]["std"]))
    out = predict(model, loaders["test"], device)
    test, pred, true = compute_metrics(out, scaler, mode, target_names)

    (run_dir / "test_metrics.json").write_text(json.dumps({"best_epoch": best_epoch, **test}, indent=2))
    with open(run_dir / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "syn_id"] + [f"{k}_{t}" for t in target_names for k in ("true", "pred")])
        for i, (d, s) in enumerate(zip(out["design"], out["syn_id"])):
            w.writerow([d, s] + [v for j in range(len(target_names)) for v in (true[i, j], pred[i, j])])
    plot_loss_curves(history, run_dir / "loss_curve.png")
    plot_pred_vs_actual(pred, true, out["design"], target_names, run_dir / "test_scatter.png")

    print("\n=== Summary ===")
    print(f"Best epoch: {best_epoch} (val loss {best_val:.4f})")
    print(summary_line("test", test, target_names))
    for t in target_names:
        print(f"  tau_{t} by design: " + ", ".join(f"{d} {v:.3f}" for d, v in test[f"tau_{t}_by_design"].items()))
    print(f"Saved to {run_dir}")
    return run_dir


def main(argv=None):
    args = parse_args(argv)
    run(load_config(args), args.runs_dir)


if __name__ == "__main__":
    main()
