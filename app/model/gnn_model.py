import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SafeRouteGNN(nn.Module):
    def __init__(self, node_features=3, edge_features=4, hidden_dim=64):
        super().__init__()

        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        src, dst = edge_index
        edge_emb = torch.cat([x[src], x[dst], edge_attr], dim=1)

        return self.edge_mlp(edge_emb).squeeze()