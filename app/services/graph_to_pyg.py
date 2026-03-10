import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np


def nx_to_pyg(graph: nx.MultiDiGraph):

    # -----------------------------------------
    # 1️⃣ Deterministic node ordering
    # -----------------------------------------
    nodes = list(graph.nodes())
    node_mapping = {n: i for i, n in enumerate(nodes)}

    num_nodes = len(nodes)

    # -----------------------------------------
    # 2️⃣ Compute node features EXACTLY like training
    # Training node features:
    # [degree, avg_crime_score, avg_light_score]
    # -----------------------------------------

    degree = np.zeros((num_nodes, 1), dtype=np.float32)
    sum_crime = np.zeros((num_nodes, 1), dtype=np.float32)
    sum_light = np.zeros((num_nodes, 1), dtype=np.float32)
    count_edges = np.zeros((num_nodes, 1), dtype=np.float32)

    # Loop through edges to accumulate stats
    for u, v, k, data in graph.edges(keys=True, data=True):

        u_idx = node_mapping[u]
        v_idx = node_mapping[v]

        crime = float(data.get("crime_score", 0.0))
        light = float(data.get("light_score", 0.0))

        degree[u_idx] += 1
        degree[v_idx] += 1

        sum_crime[u_idx] += crime
        sum_crime[v_idx] += crime

        sum_light[u_idx] += light
        sum_light[v_idx] += light

        count_edges[u_idx] += 1
        count_edges[v_idx] += 1

    # Avoid divide-by-zero
    avg_crime = sum_crime / (count_edges + 1e-9)
    avg_light = sum_light / (count_edges + 1e-9)

    node_features = np.hstack([degree, avg_crime, avg_light]).astype(np.float32)

    x = torch.tensor(node_features, dtype=torch.float)

    # -----------------------------------------
    # 3️⃣ Deterministic edge ordering
    # -----------------------------------------

    edges = list(graph.edges(keys=True))

    edge_index = []
    edge_attr = []

    for (u, v, k) in edges:

        data = graph[u][v][k]

        edge_index.append([node_mapping[u], node_mapping[v]])

        # MUST match training order exactly:
        # [light_score, crime_score, police_proximity_score, crowd_score]

        light = float(data.get("light_score", 0.0))
        crime = float(data.get("crime_score", 0.0))
        police = float(data.get("police_proximity_score", 0.0))
        crowd = float(data.get("crowd_score", 0.0))

        edge_attr.append([light, crime, police, crowd])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # -----------------------------------------
    # 4️⃣ Debug prints (extremely important)
    # -----------------------------------------

    print("\n🔎 NODE FEATURE STATS:")
    print("Degree     → min:", x[:,0].min().item(),
          "max:", x[:,0].max().item(),
          "mean:", x[:,0].mean().item())

    print("Avg Crime  → min:", x[:,1].min().item(),
          "max:", x[:,1].max().item(),
          "mean:", x[:,1].mean().item())

    print("Avg Light  → min:", x[:,2].min().item(),
          "max:", x[:,2].max().item(),
          "mean:", x[:,2].mean().item())

    print("--------------------------------------------------\n")

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return pyg_data, edges