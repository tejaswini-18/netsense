"""Smoke test: one short training run end to end (skipped without the dataset)."""

import json

from src.train import run


def test_one_epoch_run_writes_all_outputs(data_root, tmp_path):
    cfg = {
        "experiment_name": "smoke", "seed": 0,
        "data": {"root": str(data_root), "designs": ["i2c"], "split": "random", "val_fraction": 0.15,
                 "test_fraction": 0.15, "split_file": str(tmp_path / "split.json"),
                 "targets": ["and", "delay"], "target_mode": "ratio"},
        "model": {"graph_encoder": "gcn", "graph_layers": 2, "recipe_encoder": "cnn", "hidden_dim": 16},
        "training": {"lr": 1e-3, "batch_size": 64, "max_epochs": 1, "early_stopping_patience": 5},
    }
    run_dir = run(cfg, runs_dir=tmp_path / "runs")
    for name in ["config.yaml", "metrics.csv", "best.pt", "test_metrics.json",
                 "test_predictions.csv", "loss_curve.png", "test_scatter.png"]:
        assert (run_dir / name).exists(), name
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    assert metrics["best_epoch"] == 1
    assert {"mape_and", "tau_and", "mape_delay", "tau_delay"} <= set(metrics)
