"""Tests for src/evaluate.py: metrics on hand-made numbers where the right answer is known."""

import math

import numpy as np
import torch

from src.data import TargetScaler
from src.evaluate import compute_metrics, kendall_tau_per_design, mape, top_k_overlap

TRUE = np.array([10.0, 20, 30, 40, 1, 2, 3, 4])
DESIGNS = ["a"] * 4 + ["b"] * 4


def test_mape():
    assert mape(TRUE, TRUE) == 0
    assert math.isclose(mape(TRUE * 1.1, TRUE), 10.0)


def test_kendall_tau_is_computed_per_design():
    assert kendall_tau_per_design(TRUE, TRUE, DESIGNS) == {"a": 1.0, "b": 1.0}
    assert kendall_tau_per_design(-TRUE, TRUE, DESIGNS) == {"a": -1.0, "b": -1.0}
    # Ranking only: a wrong scale with the right order is still perfect
    assert kendall_tau_per_design(TRUE * 5 + 100, TRUE, DESIGNS) == {"a": 1.0, "b": 1.0}


def test_kendall_tau_skips_designs_with_constant_truth():
    true = np.array([5.0, 5, 5, 1, 2, 3])
    taus = kendall_tau_per_design(np.arange(6.0), true, ["c"] * 3 + ["d"] * 3)
    assert list(taus) == ["d"]


def test_top_k_overlap():
    assert top_k_overlap(TRUE, TRUE, DESIGNS, k=2) == {"a": 1.0, "b": 1.0}
    assert top_k_overlap(-TRUE, TRUE, DESIGNS, k=2) == {"a": 0.0, "b": 0.0}


def test_compute_metrics_undoes_scaling_and_ratio():
    # Real values 100 and 200 with base 100 -> ratios 1 and 2; standardised with mean 1.5, std 0.5
    scaler = TargetScaler(torch.tensor([1.5]), torch.tensor([0.5]))
    scaled = torch.tensor([[-1.0], [1.0]])
    out = {"pred": scaled, "y": scaled, "base": torch.tensor([[100.0], [100.0]]), "design": ["x", "x"]}
    metrics, pred, true = compute_metrics(out, scaler, "ratio", ["and"], k=1)
    assert np.allclose(true[:, 0], [100.0, 200.0])
    assert metrics["mape_and"] == 0 and metrics["tau_and"] == 1.0 and metrics["loss"] == 0
