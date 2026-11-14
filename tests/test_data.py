"""Tests for src/data.py (M3 acceptance criteria).

The first group uses small made-up inputs and always runs.
The second group needs the real dataset and is skipped without it.
"""

import pytest
import torch
from torch_geometric.loader import DataLoader

from src.data import (NUM_COMMANDS, NUM_NODE_FEATURES, NUM_RECIPES, RECIPE_LENGTH, TargetScaler,
                      build_datasets, compute_targets, load_design_graph, load_labels, load_recipes,
                      load_split, make_by_design_split, make_leave_one_design_out_splits,
                      make_random_split, save_split, to_real_scale)

DESIGNS = ["i2c", "spi", "fir", "wb_dma"]


def assert_no_overlap(split):
    parts = {name: set(samples) for name, samples in split.items()}
    for name, samples in split.items():
        assert len(samples) == len(parts[name]), f"duplicates inside {name}"
    assert not parts["train"] & parts["val"]
    assert not parts["train"] & parts["test"]
    assert not parts["val"] & parts["test"]


# ---------------------------------------------------------------------------
# Splits (no dataset needed)
# ---------------------------------------------------------------------------

def test_random_split_has_no_overlap_and_covers_everything():
    split = make_random_split(DESIGNS)
    assert_no_overlap(split)
    assert sum(len(s) for s in split.values()) == len(DESIGNS) * NUM_RECIPES


def test_random_split_holds_out_the_same_recipes_for_every_design():
    split = make_random_split(DESIGNS)
    test_ids = {d: {i for dd, i in split["test"] if dd == d} for d in DESIGNS}
    assert all(ids == test_ids["i2c"] for ids in test_ids.values())


def test_by_design_split_keeps_test_designs_out_of_training():
    split = make_by_design_split(DESIGNS, ["wb_dma"])
    assert_no_overlap(split)
    assert {d for d, _ in split["test"]} == {"wb_dma"}
    assert "wb_dma" not in {d for d, _ in split["train"] + split["val"]}
    assert len(split["test"]) == NUM_RECIPES


def test_leave_one_design_out_makes_one_fold_per_design():
    folds = make_leave_one_design_out_splits(DESIGNS)
    assert list(folds) == DESIGNS
    for held_out, split in folds.items():
        assert_no_overlap(split)
        assert {d for d, _ in split["test"]} == {held_out}


def test_splits_are_reproducible_from_the_seed():
    assert make_random_split(DESIGNS, seed=7) == make_random_split(DESIGNS, seed=7)
    assert make_random_split(DESIGNS, seed=7) != make_random_split(DESIGNS, seed=8)


def test_split_file_round_trip(tmp_path):
    split = make_by_design_split(DESIGNS, ["fir"])
    save_split(split, tmp_path / "s.json")
    assert load_split(tmp_path / "s.json") == split


# ---------------------------------------------------------------------------
# Targets and scaling (no dataset needed)
# ---------------------------------------------------------------------------

def test_ratio_targets_convert_back_to_raw():
    labels = torch.tensor([[100.0, 0, 5, 80.0, 400.0], [90.0, 0, 6, 70.0, 450.0]])
    info = {"and_nodes": 120, "longest_path": 10, "not_edges": 3}
    raw = compute_targets(labels, info, [0, 1], ["and", "delay"], "raw")
    ratio = compute_targets(labels, info, [0, 1], ["and", "delay"], "ratio")
    assert torch.equal(raw, torch.tensor([[100.0, 400.0], [90.0, 450.0]]))
    assert torch.allclose(to_real_scale(ratio, torch.tensor([120.0, 10.0]), "ratio"), raw)


def test_scaler_round_trip_and_uses_only_fitted_data():
    train = torch.tensor([[1.0, 10.0], [3.0, 30.0]])
    scaler = TargetScaler().fit(train)
    assert torch.allclose(scaler.mean, torch.tensor([2.0, 20.0]))
    other = torch.tensor([[100.0, 1000.0]])
    assert torch.allclose(scaler.inverse_transform(scaler.transform(other)), other)


# ---------------------------------------------------------------------------
# Real data (skipped when the dataset is not available)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recipes(data_root):
    return load_recipes(data_root)


def test_recipes_have_length_20_and_valid_command_ids(recipes):
    assert recipes.shape == (NUM_RECIPES, RECIPE_LENGTH)
    assert recipes.min() >= 0 and recipes.max() < NUM_COMMANDS


@pytest.mark.parametrize("design", DESIGNS)
def test_design_graph_is_valid(data_root, design):
    graph, info = load_design_graph(data_root, design)
    assert graph.x.shape == (info["num_nodes"], NUM_NODE_FEATURES)
    assert not torch.isnan(graph.x).any()
    assert graph.edge_index.min() >= 0 and graph.edge_index.max() < graph.num_nodes
    # After the flip, row 1 is the receiving gate: AND gates have exactly 2 inputs, PIs none
    fan_in = torch.bincount(graph.edge_index[1], minlength=graph.num_nodes)
    kind = graph.x[:, :3].argmax(dim=1)
    assert set(fan_in[kind == 2].tolist()) == {2}
    assert set(fan_in[kind == 0].tolist()) == {0}


def test_labels_have_no_nans(data_root):
    labels = load_labels(data_root)
    for design in DESIGNS:
        assert labels[design].shape == (NUM_RECIPES, 5)
        assert not torch.isnan(labels[design]).any()


@pytest.fixture(scope="module")
def built(data_root, tmp_path_factory):
    cfg = {"root": data_root, "designs": DESIGNS, "split": "random", "test_designs": ["wb_dma"],
           "val_fraction": 0.15, "test_fraction": 0.15, "targets": ["and", "delay"], "target_mode": "ratio",
           "split_file": tmp_path_factory.mktemp("splits") / "split.json"}
    return build_datasets(cfg, seed=42)


def test_dataloader_batch_has_expected_shapes(built):
    datasets, _, _ = built
    batch = next(iter(DataLoader(datasets["train"], batch_size=8)))
    assert batch.num_graphs == 8
    assert batch.recipe.shape == (8, RECIPE_LENGTH)
    assert batch.y.shape == (8, 2)
    assert batch.base.shape == (8, 2)
    assert batch.x.shape[1] == NUM_NODE_FEATURES
    assert len(batch.design) == 8
    assert not torch.isnan(batch.y).any()


def test_scaler_is_fitted_on_training_targets(built):
    datasets, _, _ = built
    assert torch.allclose(datasets["train"].targets.mean(dim=0), torch.zeros(2), atol=1e-4)


def test_graphs_are_shared_not_copied(built):
    datasets, _, _ = built
    first, second = datasets["train"][0], datasets["train"][1]
    if first.design == second.design:
        assert first.x.data_ptr() == second.x.data_ptr()


def test_excluded_design_is_rejected(data_root, tmp_path):
    cfg = {"root": data_root, "designs": ["i2c", "xbar"], "split": "random", "val_fraction": 0.15,
           "test_fraction": 0.15, "targets": ["and", "delay"], "target_mode": "raw",
           "split_file": tmp_path / "s.json"}
    with pytest.raises(ValueError, match="xbar"):
        build_datasets(cfg)


def test_stale_split_file_is_rejected(data_root, tmp_path):
    cfg = {"root": data_root, "designs": ["i2c", "spi"], "split": "random", "val_fraction": 0.15,
           "test_fraction": 0.15, "targets": ["and", "delay"], "target_mode": "raw",
           "split_file": tmp_path / "s.json"}
    save_split(make_random_split(["i2c", "spi"], seed=999), cfg["split_file"])
    with pytest.raises(ValueError, match="does not match"):
        build_datasets(cfg, seed=42)
