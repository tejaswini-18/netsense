"""Tests for src/run_generalisation.py (M7). The first tests use made-up data; the last needs the dataset."""

import pytest
import torch
import yaml
from torch_geometric.data import Data

from src.data import NUM_COMMANDS, NUM_NODE_FEATURES, NUM_RECIPES, RECIPE_LENGTH, QoRDataset, make_loader
from src.evaluate import predict
from src.models import build_model
from src.run_generalisation import check_folds, few_shot_ids, fine_tune, fold_config
from src.utils import set_seed


def test_folds_must_hold_out_every_design_exactly_once():
    check_folds([["a", "b"], ["c"]], ["a", "b", "c"])
    with pytest.raises(ValueError):
        check_folds([["a"], ["c"]], ["a", "b", "c"])          # b never held out
    with pytest.raises(ValueError):
        check_folds([["a", "b"], ["b", "c"]], ["a", "b", "c"])  # b held out twice


def test_fold_config_holds_out_the_fold_and_leaves_the_original_alone():
    cfg = {"experiment_name": "m7", "data": {"split": "random", "test_designs": [], "split_file": "splits/m7_random.json"}}
    fcfg = fold_config(cfg, 2, ["x", "y"])
    assert fcfg["experiment_name"] == "m7_fold2"
    assert fcfg["data"]["split"] == "by_design" and fcfg["data"]["test_designs"] == ["x", "y"]
    assert fcfg["data"]["split_file"].replace("\\", "/") == "splits/m7_fold2.json"
    assert cfg["data"]["split"] == "random" and cfg["data"]["test_designs"] == []


def test_few_shot_ids_are_fixed_and_distinct():
    ids = few_shot_ids(42, 20)
    assert ids == few_shot_ids(42, 20) and len(set(ids)) == 20
    assert all(0 <= i < NUM_RECIPES for i in ids)


def test_fine_tuning_fits_the_shots_and_keeps_the_original_model():
    set_seed(0)
    g = Data(x=torch.eye(NUM_NODE_FEATURES)[torch.randint(0, NUM_NODE_FEATURES, (10,))],
             edge_index=torch.randint(0, 10, (2, 20)), num_nodes=10)
    data = QoRDataset([("d", i) for i in range(20)], {"d": g}, torch.randint(0, NUM_COMMANDS, (20, RECIPE_LENGTH)),
                      torch.randn(20, 2), torch.ones(20, 2))
    model = build_model({"graph_encoder": "gcn", "recipe_encoder": "cnn", "hidden_dim": 16, "graph_layers": 2}, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    def shot_loss(m):
        out = predict(m, make_loader(data, batch_size=20), torch.device("cpu"))
        return torch.nn.functional.mse_loss(out["pred"], out["y"]).item()

    tuned = fine_tune(model, data, {"loss": "mse"}, {"epochs": 50, "lr": 1e-2}, torch.device("cpu"), seed=0)
    assert shot_loss(tuned) < shot_loss(model)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # original untouched


def test_runner_produces_all_settings(data_root, tmp_path):
    from src.run_generalisation import main
    designs = ["i2c", "spi", "fir", "wb_dma"]
    cfg = {"experiment_name": "tiny", "seed": 0,
           "data": {"root": str(data_root), "designs": designs, "split": "random", "test_designs": [],
                    "val_fraction": 0.15, "test_fraction": 0.15, "split_seed": 42,
                    "split_file": str(tmp_path / "splits" / "m7_random.json"),
                    "targets": ["and", "delay"], "target_mode": "ratio"},
           "model": {"graph_encoder": "gcn", "graph_layers": 2, "recipe_encoder": "cnn", "hidden_dim": 16},
           "training": {"lr": 1e-3, "batch_size": 64, "max_epochs": 1, "early_stopping_patience": 5}}
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump(cfg))
    plan = {"variants": [str(tmp_path / "tiny.yaml")], "seeds": [0],
            "folds": [["i2c", "spi"], ["fir", "wb_dma"]], "few_shot": {"shots": 20, "epochs": 2, "lr": 1e-3}}
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan))
    out = tmp_path / "results"
    main(["--plan", str(tmp_path / "plan.yaml"), "--runs-dir", str(tmp_path / "runs"), "--out", str(out)])

    rows = (out / "all_runs.csv").read_text().strip().splitlines()
    assert [r.split(",")[1] for r in rows[1:]] == ["random", "unseen", "few_shot"]
    per_design = (out / "per_design.csv").read_text().strip().splitlines()
    assert len(per_design) == 1 + 3 * len(designs)       # every design scored in every setting
    table = (out / "results_table.md").read_text(encoding="utf-8")
    assert table.count("reference:") == 3
    assert (out / "generalisation_bar.png").exists()
