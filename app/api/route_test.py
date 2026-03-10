from fastapi import APIRouter, Request
from app.services.route_engine import RouteEngine

router = APIRouter(prefix="/test", tags=["Route Engine Tests"])

@router.get("/route")
def test_routes(request: Request, mode: str = "walk"):
    graph = request.app.state.road_graphs[mode]
    engine = RouteEngine(graph)

    nodes = list(graph.nodes)
    start = nodes[10]
    end = nodes[500]

    return {
        "shortest": engine.build_route_response(
            engine.shortest_route(start, end),
            mode,
            "shortest"
        ),
        "safest": engine.build_route_response(
            engine.safest_route(start, end),
            mode,
            "safest"
        ),
        "optimal": engine.build_route_response(
            engine.optimal_route(start, end),
            mode,
            "optimal"
        )
    }
