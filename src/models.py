"""Graph encoders, recipe encoders and the full QoR model (PRD sections 7.3 and 7.4).

Input:  a batch from src.data.QoRDataset (graphs + recipes)
Output: [batch_size, num_targets] standardised predictions (e.g. AND count and delay)
Every choice is made in the 'model' section of a config (see build_model).
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool

from src.data import NUM_COMMANDS, NUM_NODE_FEATURES, RECIPE_LENGTH


def make_conv(kind, in_dim, out_dim):
    """One message-passing layer of the chosen type."""
    if kind == "gcn":
        # Weighted average of a node and its neighbours (degree-normalised)
        return GCNConv(in_dim, out_dim)
    if kind == "sage":
        # Keeps the node's own vector separate from the mean of its neighbours
        return SAGEConv(in_dim, out_dim)
    if kind == "gin":
        # Sum of the node and its neighbours, passed through a small MLP
        return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
    raise ValueError(f"unknown graph_encoder {kind!r} (use gcn, sage or gin)")


class GraphEncoder(nn.Module):
    """Stack of message-passing layers, then mean pooling -> one vector per graph."""

    def __init__(self, in_dim=NUM_NODE_FEATURES, hidden_dim=64, num_layers=3, kind="gcn"):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(make_conv(kind, dims[i], dims[i + 1]) for i in range(num_layers))

    def forward(self, x, edge_index, batch):
        # Each layer: every node mixes its own vector with those of the nodes wired into it
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        # Average all node vectors of each graph -> [num_graphs, hidden_dim], independent of circuit size
        return global_mean_pool(x, batch)


class RecipeEncoder(nn.Module):
    """Embedding per ABC command, 1D CNN over the 20 steps, then mean pooling -> one vector per recipe."""

    def __init__(self, num_commands=NUM_COMMANDS, embed_dim=32, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.embed = nn.Embedding(num_commands, embed_dim)
        pad = kernel_size // 2  # keep the sequence length at 20
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad)

    def forward(self, recipe):
        # recipe: [B, 20] command IDs -> [B, 20, embed_dim]
        h = self.embed(recipe)
        # Conv1d expects [B, channels, length], so move the embedding dimension to the middle
        h = h.transpose(1, 2)
        h = torch.relu(self.conv1(h))
        h = torch.relu(self.conv2(h))
        # Average over the 20 steps -> [B, hidden_dim]
        return h.mean(dim=2)


class TransformerRecipeEncoder(nn.Module):
    """Embedding + learned positional embedding, small transformer encoder, mean pooling.

    Self-attention lets every step look at every other step; the positional embedding tells it the order.
    """

    def __init__(self, num_commands=NUM_COMMANDS, hidden_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(num_commands, hidden_dim)
        self.position = nn.Embedding(RECIPE_LENGTH, hidden_dim)  # one learned vector per step 0..19
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=2 * hidden_dim,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, recipe):
        positions = torch.arange(recipe.shape[1], device=recipe.device)
        h = self.embed(recipe) + self.position(positions)   # [B, 20, hidden_dim]
        return self.encoder(h).mean(dim=1)                   # [B, hidden_dim]


class QoRModel(nn.Module):
    """Graph vector + recipe vector -> 2-layer MLP -> predicted targets.

    inputs = "graph_only" or "recipe_only" drops one branch (ablation study).
    """

    def __init__(self, num_targets=2, hidden_dim=64, graph_layers=3, graph_encoder="gcn",
                 recipe_encoder="cnn", inputs="both"):
        super().__init__()
        if inputs not in ("both", "graph_only", "recipe_only"):
            raise ValueError(f"unknown inputs {inputs!r} (use both, graph_only or recipe_only)")
        self.inputs = inputs
        self.graph_encoder = None
        self.recipe_encoder = None
        if inputs != "recipe_only":
            self.graph_encoder = GraphEncoder(hidden_dim=hidden_dim, num_layers=graph_layers, kind=graph_encoder)
        if inputs != "graph_only":
            if recipe_encoder == "cnn":
                self.recipe_encoder = RecipeEncoder(hidden_dim=hidden_dim)
            elif recipe_encoder == "transformer":
                self.recipe_encoder = TransformerRecipeEncoder(hidden_dim=hidden_dim)
            else:
                raise ValueError(f"unknown recipe_encoder {recipe_encoder!r} (use cnn or transformer)")
        branches = 2 if inputs == "both" else 1
        self.head = nn.Sequential(
            nn.Linear(branches * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, batch):
        parts = []
        if self.graph_encoder is not None:
            parts.append(self.graph_encoder(batch.x, batch.edge_index, batch.batch))
        if self.recipe_encoder is not None:
            parts.append(self.recipe_encoder(batch.recipe))
        return self.head(torch.cat(parts, dim=1))


def build_model(model_cfg, num_targets):
    """Create the model from the 'model' section of a config."""
    return QoRModel(num_targets=num_targets, hidden_dim=model_cfg["hidden_dim"],
                    graph_layers=model_cfg["graph_layers"], graph_encoder=model_cfg["graph_encoder"],
                    recipe_encoder=model_cfg["recipe_encoder"], inputs=model_cfg.get("inputs", "both"))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
