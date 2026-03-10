import torch
import networkx as nx
from app.services.graph_to_pyg import nx_to_pyg

class GNNGraphInjector:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def inject(self, graph: nx.MultiDiGraph):

        # Get deterministic PyG data + ordered edge list
        pyg_data, edges = nx_to_pyg(graph)

        scores = self.model(
            pyg_data.x,
            pyg_data.edge_index,
            pyg_data.edge_attr
        )

        # Convert logits → safety probability
        #scores = torch.sigmoid(scores)

        scores = scores.cpu()

        # Normalize to 0-1 using min-max scaling
        min_val = scores.min()
        max_val = scores.max()

        scores = (scores - min_val) / (max_val - min_val + 1e-8)

        scores = scores.numpy()

        print(
            "GNN SCORE STATS:",
            "min=", scores.min(),
            "max=", scores.max(),
            "mean=", scores.mean()
        )
        import numpy as np

        print("Percentiles:",
            np.percentile(scores, [5, 25, 50, 75, 95]))
        print("Sample edge_attr row:", pyg_data.edge_attr[0])

        # ✅ Use SAME edge ordering returned by nx_to_pyg
        for i, (u, v, k) in enumerate(edges):

            safety_score = float(scores[i])

            # Clamp
            safety_score = max(0.0, min(1.0, safety_score))

            # Convert safety → cost for Dijkstra
            graph[u][v][k]["risk_weight"] = 1.0 - safety_score

        return graph