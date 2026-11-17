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
import math
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.data import TargetScaler, build_datasets, make_loader
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


def pairwise_ranking_loss(pred, target, groups):
    """Logistic pairwise loss over pairs of samples from the same design (RankNet-style).

    For every pair (i, j) of the same design whose true values differ, the loss is small when the predicted
    difference has the same sign as the true difference: log(1 + exp(-sign(t_i - t_j) * (p_i - p_j))).
    Computed per target column and averaged over all valid pairs.
    """
    same_design = groups[:, None] == groups[None, :]              # [B, B]
    true_diff = target[:, None, :] - target[None, :, :]          # [B, B, T]
    pred_diff = pred[:, None, :] - pred[None, :, :]
    valid = same_design[..., None] & (true_diff != 0)
    if not valid.any():
        return pred.sum() * 0.0  # no usable pair in this batch
    losses = torch.nn.functional.softplus(-torch.sign(true_diff) * pred_diff)
    return losses[valid].mean()


def compute_loss(pred, batch, train_cfg):
    """MSE, plus the pairwise ranking loss when train_cfg['loss'] == 'mse_rank'."""
    loss = torch.nn.functional.mse_loss(pred, batch.y)
    kind = train_cfg.get("loss", "mse")
    if kind == "mse_rank":
        names = sorted(set(batch.design))
        groups = torch.tensor([names.index(d) for d in batch.design], device=pred.device)
        loss = loss + train_cfg.get("rank_weight", 1.0) * pairwise_ranking_loss(pred, batch.y, groups)
    elif kind != "mse":
        raise ValueError(f"unknown loss {kind!r} (use mse or mse_rank)")
    return loss


def selection_score(val, target_names, select_by):
    """Lower is better. val_loss: validation MSE; val_tau: minus the mean Kendall's tau over targets."""
    if select_by == "val_loss":
        return val["loss"]
    if select_by == "val_tau":
        taus = [val[f"tau_{t}"] for t in target_names if not math.isnan(val[f"tau_{t}"])]
        return -sum(taus) / len(taus) if taus else float("inf")
    raise ValueError(f"unknown select_by {select_by!r} (use val_loss or val_tau)")


def train_one_epoch(model, loader, optimizer, device, train_cfg):
    """One pass over the training data; returns the average training loss."""
    model.train()
    total, count = 0.0, 0
    for batch in tqdm(loader, desc="  train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = compute_loss(model(batch), batch, train_cfg)
        loss.backward()           # compute gradients
        optimizer.step()          # update the weights
        total += loss.item() * len(batch)
        count += len(batch)
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

    # The split uses its own seed, so changing the training seed keeps the same train/val/test samples
    datasets, scaler, meta = build_datasets(data_cfg, data_cfg.get("split_seed", cfg["seed"]))
    print(f"Samples: {meta['split_sizes']}")
    # The training order is shuffled reproducibly from the seed
    loaders = {
        "train": make_loader(datasets["train"], train_cfg["batch_size"], shuffle=True, seed=cfg["seed"]),
        "val": make_loader(datasets["val"], train_cfg["batch_size"]),
        "test": make_loader(datasets["test"], train_cfg["batch_size"]),
    }

    model = build_model(cfg["model"], num_targets=len(target_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    print(f"Model parameters: {count_parameters(model):,}")

    select_by = train_cfg.get("select_by", "val_loss")
    history, best_score, best_val, best_epoch, waited = [], float("inf"), float("inf"), 0, 0
    columns = ["epoch", "train_loss", "val_loss"] + [f"val_{m}_{t}" for t in target_names
                                                     for m in ("mape", "tau", "top10")] + ["seconds"]
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(columns)

    for epoch in range(1, train_cfg["max_epochs"] + 1):
        start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device, train_cfg)
        val, _, _ = compute_metrics(predict(model, loaders["val"], device), scaler, mode, target_names)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"], "seconds": time.time() - start}
        for t in target_names:
            for m in ("mape", "tau", "top10"):
                row[f"val_{m}_{t}"] = val[f"{m}_{t}"]
        history.append(row)
        with open(run_dir / "metrics.csv", "a", newline="") as f:
            csv.writer(f).writerow([row[c] for c in columns])

        # Early stopping: keep the best model by the chosen validation score, stop when it stops improving
        score = selection_score(val, target_names, select_by)
        improved = score < best_score
        if improved:
            best_score, best_val, best_epoch, waited = score, val["loss"], epoch, 0
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
    print(f"Best epoch: {best_epoch} (chosen by {select_by}; val loss {best_val:.4f})")
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
