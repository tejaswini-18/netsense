"""Tests for src/models.py (M4 acceptance criteria), using small made-up graphs so they run anywhere."""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.data import NUM_COMMANDS, NUM_NODE_FEATURES, RECIPE_LENGTH
from src.models import GraphEncoder, QoRModel, RecipeEncoder, build_model
from src.utils import set_seed

MODEL_CFG = {"graph_encoder": "gcn", "recipe_encoder": "cnn", "hidden_dim": 32, "graph_layers": 3}


def fake_sample(num_nodes=12):
    """A small random graph shaped like a real sample from QoRDataset."""
    x = torch.zeros(num_nodes, NUM_NODE_FEATURES)
    x[torch.arange(num_nodes), torch.randint(0, 3, (num_nodes,))] = 1           # node type one-hot
    x[torch.arange(num_nodes), 3 + torch.randint(0, 3, (num_nodes,))] = 1       # inverted-input one-hot
    edge_index = torch.randint(0, num_nodes, (2, 2 * num_nodes))
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes,
                recipe=torch.randint(0, NUM_COMMANDS, (1, RECIPE_LENGTH)), y=torch.randn(1, 2))


def fake_batch(n):
    return next(iter(DataLoader([fake_sample(num_nodes=8 + i % 5) for i in range(n)], batch_size=n)))


def test_encoders_output_one_vector_per_sample():
    batch = fake_batch(6)
    assert GraphEncoder(hidden_dim=32)(batch.x, batch.edge_index, batch.batch).shape == (6, 32)
    assert RecipeEncoder(hidden_dim=32)(batch.recipe).shape == (6, 32)


def test_forward_pass_gives_batch_by_2():
    model = build_model(MODEL_CFG, num_targets=2)
    out = model(fake_batch(8))
    assert out.shape == (8, 2)
    assert not torch.isnan(out).any()


def test_graph_vector_does_not_depend_on_graph_size():
    # Mean pooling: duplicating every node and edge of a graph leaves its vector unchanged
    set_seed(0)
    g = fake_sample(10)
    doubled = Data(x=torch.cat([g.x, g.x]),
                   edge_index=torch.cat([g.edge_index, g.edge_index + 10], dim=1), num_nodes=20)
    enc = GraphEncoder(hidden_dim=16)
    one = enc(g.x, g.edge_index, torch.zeros(10, dtype=torch.long))
    two = enc(doubled.x, doubled.edge_index, torch.zeros(20, dtype=torch.long))
    assert torch.allclose(one, two, atol=1e-5)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        build_model({**MODEL_CFG, "graph_encoder": "gat"}, num_targets=2)
    with pytest.raises(ValueError):
        build_model({**MODEL_CFG, "recipe_encoder": "lstm"}, num_targets=2)
    with pytest.raises(ValueError):
        build_model({**MODEL_CFG, "inputs": "neither"}, num_targets=2)


@pytest.mark.parametrize("change", [{"graph_encoder": "gin"}, {"graph_encoder": "sage"},
                                    {"recipe_encoder": "transformer"}, {"inputs": "graph_only"},
                                    {"inputs": "recipe_only"}])
def test_every_variant_gives_batch_by_2(change):
    model = build_model({**MODEL_CFG, **change}, num_targets=2)
    out = model(fake_batch(6))
    assert out.shape == (6, 2)
    assert not torch.isnan(out).any()


def test_ablations_really_drop_a_branch():
    assert build_model({**MODEL_CFG, "inputs": "graph_only"}, 2).recipe_encoder is None
    assert build_model({**MODEL_CFG, "inputs": "recipe_only"}, 2).graph_encoder is None


def test_model_can_overfit_50_samples():
    # Sanity check: if the model cannot memorise 50 samples, something in the plumbing is broken
    set_seed(0)
    batch = fake_batch(50)
    model = QoRModel(num_targets=2, hidden_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    start = torch.nn.functional.mse_loss(model(batch), batch.y).item()
    for _ in range(500):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(batch), batch.y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.01 * start
