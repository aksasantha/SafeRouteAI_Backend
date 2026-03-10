# app/services/gnn_schema.py

# Node features (x)
NODE_FEATURE_KEYS = [
    "lat",
    "lon",
    "degree"
]

# Edge features (edge_attr) — ORDER MATTERS
EDGE_FEATURE_KEYS = [
    "light_score",
    "crime_score",
    "police_score",
    "crowd_score"
]

NUM_NODE_FEATURES = 3
NUM_EDGE_FEATURES = 4