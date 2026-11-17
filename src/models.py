"""Graph encoder, recipe encoder and the full QoR model (PRD section 7.3).

Input:  a batch from src.data.QoRDataset (graphs + recipes)
Output: [batch_size, num_targets] standardised predictions (e.g. AND count and delay)
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from src.data import NUM_COMMANDS, NUM_NODE_FEATURES


class GraphEncoder(nn.Module):
    """Stack of GCN layers (message passing), then mean pooling -> one vector per graph."""

    def __init__(self, in_dim=NUM_NODE_FEATURES, hidden_dim=64, num_layers=3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(num_layers))

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


class QoRModel(nn.Module):
    """Graph vector + recipe vector -> 2-layer MLP -> predicted targets."""

    def __init__(self, num_targets=2, hidden_dim=64, graph_layers=3):
        super().__init__()
        self.graph_encoder = GraphEncoder(hidden_dim=hidden_dim, num_layers=graph_layers)
        self.recipe_encoder = RecipeEncoder(hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, batch):
        g = self.graph_encoder(batch.x, batch.edge_index, batch.batch)
        r = self.recipe_encoder(batch.recipe)
        return self.head(torch.cat([g, r], dim=1))


def build_model(model_cfg, num_targets):
    """Create the model from the 'model' section of a config."""
    if model_cfg["graph_encoder"] != "gcn" or model_cfg["recipe_encoder"] != "cnn":
        raise ValueError("only graph_encoder=gcn and recipe_encoder=cnn exist so far (variants come in M6)")
    return QoRModel(num_targets=num_targets, hidden_dim=model_cfg["hidden_dim"],
                    graph_layers=model_cfg["graph_layers"])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
