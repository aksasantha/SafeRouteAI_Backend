from fastapi import APIRouter
from app.services.gnn_loader import GNNModelLoader
from app.schemas.graph_input import GraphInput

router = APIRouter(prefix="/predict", tags=["Prediction"])

gnn_loader = GNNModelLoader()

@router.post("/route")
def predict_route(graph: GraphInput):
    """
    Receives a route graph and returns edge safety scores
    """
    data = graph.to_pyg()
    scores = gnn_loader.predict(data)

    return {
        "num_edges": len(scores),
        "edge_scores":
         scores.tolist()
    }
