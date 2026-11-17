"""Metrics (PRD section 8.1) and running a model over a dataset.

All metrics are computed on real-scale values (actual AND counts / delays),
after undoing standardisation and, for ratio targets, the division by the initial graph's value.
"""

import math

import numpy as np
import torch
from scipy.stats import kendalltau

from src.data import to_real_scale


@torch.no_grad()
def predict(model, loader, device):
    """Run the model over a DataLoader.

    Returns standardised predictions and targets plus what is needed to convert them back to real units.
    """
    model.eval()
    preds, ys, bases, designs, syn_ids = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu())
        ys.append(batch.y.cpu())
        bases.append(batch.base.cpu())
        designs.extend(batch.design)
        syn_ids.extend(batch.syn_id.cpu().tolist())
    return {"pred": torch.cat(preds), "y": torch.cat(ys), "base": torch.cat(bases),
            "design": designs, "syn_id": syn_ids}


def to_real(scaled, base, scaler, mode):
    """Standardised values -> real units (e.g. AND count)."""
    return to_real_scale(scaler.inverse_transform(scaled), base, mode)


def mape(pred, true):
    """Mean absolute percentage error, in %."""
    return float(np.mean(np.abs(pred - true) / np.abs(true)) * 100)


def kendall_tau_per_design(pred, true, designs):
    """Kendall's tau between predicted and true values, separately for each design.

    Tau measures ranking agreement: +1 same order, 0 unrelated, -1 reversed.
    Designs where tau is undefined (e.g. every true value equal) are left out.
    """
    taus = {}
    designs = np.asarray(designs)
    for d in sorted(set(designs)):
        mask = designs == d
        if mask.sum() < 2:
            continue
        tau = kendalltau(pred[mask], true[mask]).statistic
        if not math.isnan(tau):
            taus[str(d)] = float(tau)
    return taus


def top_k_overlap(pred, true, designs, k=10):
    """Per design: fraction of the k recipes with the lowest predicted value that are truly in the lowest k."""
    overlaps = {}
    designs = np.asarray(designs)
    for d in sorted(set(designs)):
        idx = np.flatnonzero(designs == d)
        if len(idx) < k:
            continue
        best_pred = set(idx[np.argsort(pred[idx], kind="stable")[:k]])
        best_true = set(idx[np.argsort(true[idx], kind="stable")[:k]])
        overlaps[str(d)] = len(best_pred & best_true) / k
    return overlaps


def compute_metrics(out, scaler, mode, target_names, k=10):
    """All metrics for one prediction run, as a flat dict (easy to write to CSV/JSON).

    Keys: loss (MSE on standardised values), and per target t: mape_t, tau_t, top{k}_t,
    plus per-design tau in tau_t_by_design.
    """
    metrics = {"loss": float(torch.nn.functional.mse_loss(out["pred"], out["y"]))}
    pred = to_real(out["pred"], out["base"], scaler, mode).numpy()
    true = to_real(out["y"], out["base"], scaler, mode).numpy()
    for j, t in enumerate(target_names):
        taus = kendall_tau_per_design(pred[:, j], true[:, j], out["design"])
        tops = top_k_overlap(pred[:, j], true[:, j], out["design"], k)
        metrics[f"mape_{t}"] = mape(pred[:, j], true[:, j])
        metrics[f"tau_{t}"] = float(np.mean(list(taus.values()))) if taus else float("nan")
        metrics[f"top{k}_{t}"] = float(np.mean(list(tops.values()))) if tops else float("nan")
        metrics[f"tau_{t}_by_design"] = taus
    return metrics, pred, true
