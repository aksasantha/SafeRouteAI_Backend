from pydantic import BaseModel
from typing import List
import torch
from torch_geometric.data import Data

class GraphInput(BaseModel):
    node_features: List[List[float]]
    edge_index: List[List[int]]
    edge_features: List[List[float]]

    def to_pyg(self) -> Data:
        x = torch.tensor(self.node_features, dtype=torch.float)
        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(self.edge_features, dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
