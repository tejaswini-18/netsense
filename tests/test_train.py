"""Tests for src/train.py and src/run_experiments.py.

The loss and selection tests use made-up numbers; the run tests need the dataset and are skipped without it.
"""

import json

import pytest
import torch
import yaml

from src.train import pairwise_ranking_loss, run, selection_score


# ---------------------------------------------------------------------------
# Ranking loss and model selection (no dataset needed)
# ---------------------------------------------------------------------------

Y = torch.tensor([[1.0], [2.0], [3.0], [10.0], [20.0]])
GROUPS = torch.tensor([0, 0, 0, 1, 1])


def test_ranking_loss_prefers_the_true_order():
    assert pairwise_ranking_loss(Y * 5, Y, GROUPS) < pairwise_ranking_loss(-Y * 5, Y, GROUPS)


def test_ranking_loss_ignores_pairs_from_different_designs():
    moved = Y.clone()
    moved[3:] -= 100  # shift design 1 far below design 0; only within-design order matters
    assert torch.isclose(pairwise_ranking_loss(moved, Y, GROUPS), pairwise_ranking_loss(Y, Y, GROUPS))


def test_ranking_loss_without_valid_pairs_is_zero():
    assert pairwise_ranking_loss(Y[:2], Y[:2], torch.tensor([0, 1])).item() == 0


def test_selection_score():
    val = {"loss": 0.5, "tau_and": 0.2, "tau_delay": 0.4}
    assert selection_score(val, ["and", "delay"], "val_loss") == 0.5
    assert selection_score(val, ["and", "delay"], "val_tau") == pytest.approx(-0.3)  # lower is better
    with pytest.raises(ValueError):
        selection_score(val, ["and"], "val_mape")


# ---------------------------------------------------------------------------
# Real runs (skipped without the dataset)
# ---------------------------------------------------------------------------

def small_config(data_root, tmp_path, **training):
    return {
        "experiment_name": "smoke", "seed": 0,
        "data": {"root": str(data_root), "designs": ["i2c"], "split": "random", "val_fraction": 0.15,
                 "test_fraction": 0.15, "split_seed": 42, "split_file": str(tmp_path / "split.json"),
                 "targets": ["and", "delay"], "target_mode": "ratio"},
        "model": {"graph_encoder": "gcn", "graph_layers": 2, "recipe_encoder": "cnn", "hidden_dim": 16},
        "training": {"lr": 1e-3, "batch_size": 64, "max_epochs": 1, "early_stopping_patience": 5, **training},
    }


def test_one_epoch_run_writes_all_outputs(data_root, tmp_path):
    run_dir = run(small_config(data_root, tmp_path), runs_dir=tmp_path / "runs")
    for name in ["config.yaml", "metrics.csv", "best.pt", "test_metrics.json",
                 "test_predictions.csv", "loss_curve.png", "test_scatter.png"]:
        assert (run_dir / name).exists(), name
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    assert metrics["best_epoch"] == 1
    assert {"mape_and", "tau_and", "mape_delay", "tau_delay"} <= set(metrics)


def test_ranking_loss_and_tau_selection_run(data_root, tmp_path):
    run_dir = run(small_config(data_root, tmp_path, loss="mse_rank", select_by="val_tau"),
                  runs_dir=tmp_path / "runs")
    assert (run_dir / "test_metrics.json").exists()


def test_runner_builds_table_and_keeps_the_split_fixed_across_seeds(data_root, tmp_path):
    from src.run_experiments import main
    configs = []
    for name, inputs in [("a_both", "both"), ("b_recipe_only", "recipe_only")]:
        cfg = small_config(data_root, tmp_path)
        cfg["experiment_name"] = name
        cfg["model"]["inputs"] = inputs
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        configs.append(str(path))
    out = tmp_path / "results"
    # Two seeds share one split_file: a seed-dependent split would make the second run fail its consistency check
    main(["--configs", *configs, "--seeds", "0", "1", "--runs-dir", str(tmp_path / "runs"), "--out", str(out)])
    rows = (out / "all_runs.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 4  # header + 2 configs x 2 seeds
    table = (out / "results_table.md").read_text()
    assert "a_both" in table and "b_recipe_only" in table and "design average" in table
    assert (out / "experiments_bar.png").exists()
