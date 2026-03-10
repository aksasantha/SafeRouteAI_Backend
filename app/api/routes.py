from fastapi import APIRouter, Request, HTTPException
from app.services.route_engine import RouteEngine

router = APIRouter(prefix="/routes", tags=["Routing"])


def percent_diff(new, base):
    if base == 0:
        return 0.0
    return ((new - base) / base) * 100


def build_quantitative_explanation(route_summary, reference_summary, route_type):
    if route_type == "shortest":
        return "This is the shortest available route by distance."

    risk_change = percent_diff(
        route_summary["avg_risk"],
        reference_summary["avg_risk"]
    )

    distance_change = percent_diff(
        route_summary["distance_m"],
        reference_summary["distance_m"]
    )

    risk_improvement = -risk_change  # lower risk is better

    explanation_parts = []

    if risk_improvement > 1:
        explanation_parts.append(
            f"reduces risk by {risk_improvement:.1f}%"
        )
    elif risk_improvement < -1:
        explanation_parts.append(
            f"increases risk by {abs(risk_improvement):.1f}%"
        )

    if abs(distance_change) > 1:
        if distance_change > 0:
            explanation_parts.append(
                f"increases distance by {distance_change:.1f}%"
            )
        else:
            explanation_parts.append(
                f"reduces distance by {abs(distance_change):.1f}%"
            )

    if not explanation_parts:
        return "This route provides a similar safety-distance balance as the shortest option."

    return (
        "This route "
        + " and ".join(explanation_parts)
        + " compared to the shortest route."
    )


@router.post("/compute")
def compute_route(payload: dict, request: Request):
    mode = payload.get("mode", "drive")

    if mode not in ("drive", "walk"):
        raise HTTPException(status_code=400, detail="Invalid mode")

    graph = request.app.state.road_graphs[mode]
    engine = RouteEngine(graph)

    paths = engine.compute_routes(
        payload["src_lat"],
        payload["src_lon"],
        payload["dst_lat"],
        payload["dst_lon"]
    )

    # Build responses first
    shortest = engine.build_route_response(
        path=paths["shortest"],
        mode=mode,
        route_type="shortest",
    )

    safest = engine.build_route_response(
        path=paths["safest"],
        mode=mode,
        route_type="safest",
    )

    optimal = engine.build_route_response(
        path=paths["optimal"],
        mode=mode,
        route_type="optimal",
    )

    # Reference summary = shortest
    ref_summary = shortest["summary"]

    # Attach explanations
    shortest["explanation"] = build_quantitative_explanation(
        shortest["summary"], ref_summary, "shortest"
    )

    safest["explanation"] = build_quantitative_explanation(
        safest["summary"], ref_summary, "safest"
    )

    optimal["explanation"] = build_quantitative_explanation(
        optimal["summary"], ref_summary, "optimal"
    )

    return {
        "shortest": shortest,
        "safest": safest,
        "optimal": optimal,
    }