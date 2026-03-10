import networkx as nx
import osmnx as ox
from typing import Dict, List
import math

class RouteEngine:
    def __init__(self, graph):
        """
        graph: NetworkX MultiDiGraph
        """
        self.graph = graph

    def _nearest_node(self, lat: float, lon: float) -> int:
        return ox.distance.nearest_nodes(self.graph, lon, lat)
    
    def _normalized_edge_cost(self, u, v, edge_dict, alpha):

        min_cost = float("inf")

        for key, d in edge_dict.items():

            length = d.get("length", 1.0)
            risk = d.get("risk_weight", 1.0)

            risk_scaled = risk * 50.0

            cost = alpha * risk_scaled + (1 - alpha) * length

            min_cost = min(min_cost, cost)

        return min_cost
        
    def _debug_path_risk(self, path, label):
        risks = []

        for u, v in zip(path[:-1], path[1:]):
            edge_data = min(
                self.graph.get_edge_data(u, v).values(),
                key=lambda d: d.get("length", 1)
            )
            risks.append(edge_data.get("risk_weight", 0))

        print(
            f"{label}: "
            f"avg={sum(risks)/len(risks):.3f}, "
            f"min={min(risks):.3f}, "
            f"max={max(risks):.3f}"
        )

    def shortest_route(self, src, dst):
        return nx.shortest_path(
            self.graph,
            src,
            dst,
            weight="length"
        )

    def safest_route(self, src, dst):
        return nx.shortest_path(
            self.graph,
            src,
            dst,
            weight=lambda u, v, d: d.get("risk_weight", 1.0)
        )

    def optimal_route(self, src, dst, alpha=0.3):
        """
        Tradeoff between safety and distance
        """
        return nx.shortest_path(
            self.graph,
            src,
            dst,
            weight=lambda u, v, d: self._normalized_edge_cost(u, v, d, alpha)
        )

    def compute_routes(
        self,
        src_lat: float,
        src_lon: float,
        dst_lat: float,
        dst_lon: float
    ) -> Dict:

        src_node = self._nearest_node(src_lat, src_lon)
        dst_node = self._nearest_node(dst_lat, dst_lon)

        shortest = self.shortest_route(src_node, dst_node)
        safest = self.safest_route(src_node, dst_node)
        optimal = self.optimal_route(src_node, dst_node)

        self._debug_path_risk(shortest, "SHORTEST")
        self._debug_path_risk(safest, "SAFEST")
        self._debug_path_risk(optimal, "OPTIMAL")

        print("PATH COMPARISON")
        print("Shortest vs Safest:", len(set(shortest) ^ set(safest)))
        print("Safest vs Optimal:", len(set(safest) ^ set(optimal)))
        print("Shortest vs Optimal:", len(set(shortest) ^ set(optimal)))

        return {
            "shortest": shortest,
            "safest": safest,
            "optimal": optimal
        }
    def summarize_route(self, path):
        total_length = 0.0
        total_risk = 0.0

        total_light = 0.0
        total_crime = 0.0
        total_police = 0.0
        total_crowd = 0.0

        edge_count = 0

        for u, v in zip(path[:-1], path[1:]):
            edge_data = min(
                self.graph.get_edge_data(u, v).values(),
                key=lambda d: d.get("length", 1)
            )

            total_length += edge_data.get("length", 0)
            total_risk += edge_data.get("risk_weight", 0)

            # 🔥 NEW FEATURE BREAKDOWN
            total_light += edge_data.get("light_score", 0)
            total_crime += edge_data.get("crime_score", 0)
            total_police += edge_data.get("police_proximity_score", 0)
            total_crowd += edge_data.get("crowd_score", 0)

            edge_count += 1
            print(edge_data)

        if edge_count == 0:
            return {
                "distance_m": 0,
                "avg_risk": 0,
                "avg_light": 0,
                "avg_crime": 0,
                "avg_police": 0,
                "avg_crowd": 0,
                "num_edges": 0
            }
        

        return {
            "distance_m": round(total_length, 2),
            "avg_risk": round(total_risk / edge_count, 3),
            "avg_light": round(total_light / edge_count, 3),
            "avg_crime": round(total_crime / edge_count, 3),
            "avg_police": round(total_police / edge_count, 3),
            "avg_crowd": round(total_crowd / edge_count, 3),
            "num_edges": edge_count
        }
    
    def path_to_geometry(self, path):
        """
        Convert node path → list of lat/lon points
        """
        coords = []

        for node in path:
            data = self.graph.nodes[node]
            coords.append({
                "lat": data["y"],
                "lon": data["x"]
            })

        return coords

    def build_route_response(self, path, mode, route_type):
        return {
            "mode": mode,
            "route_type": route_type,
            "summary": self.summarize_route(path),
            "geometry": self.path_to_geometry(path),
            "instructions": self.generate_turn_instructions(path)
        }
    
    def debug_risk_stats(self, path, name):
        risks = []
        for u, v in zip(path[:-1], path[1:]):
            edge = min(
                self.graph.get_edge_data(u, v).values(),
                key=lambda d: d.get("length", 1)
            )
            risks.append(edge.get("risk_weight", 0.0))
        print(f"{name}: avg={sum(risks)/len(risks):.3f}, min={min(risks):.3f}, max={max(risks):.3f}")
    
    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Returns bearing in degrees (0–360)
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)

        x = math.sin(delta_lon) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - \
            math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    def _classify_turn(self, angle_diff):
        """
        Classify turn based on angle difference
        """
        if angle_diff < 15:
            return None  # straight

        if angle_diff < 45:
            return "slight"

        if angle_diff < 120:
            return "left" if angle_diff > 0 else "right"

        return "sharp"
    
    def generate_turn_instructions(self, path):
        """
        Generate turn-by-turn instructions with distance to each maneuver
        """
        instructions = []

        if len(path) < 3:
            return instructions

        coords = [
            (self.graph.nodes[n]["y"], self.graph.nodes[n]["x"])
            for n in path
        ]

        cumulative_distance = 0
        last_instruction_index = 0

        for i in range(1, len(coords) - 1):

            # --- Get edge distance from previous segment ---
            edge_data_prev = min(
                self.graph.get_edge_data(path[i - 1], path[i]).values(),
                key=lambda d: d.get("length", 1)
            )

            cumulative_distance += edge_data_prev.get("length", 0)

            lat1, lon1 = coords[i - 1]
            lat2, lon2 = coords[i]
            lat3, lon3 = coords[i + 1]

            bearing1 = self._calculate_bearing(lat1, lon1, lat2, lon2)
            bearing2 = self._calculate_bearing(lat2, lon2, lat3, lon3)

            angle = bearing2 - bearing1
            angle = (angle + 180) % 360 - 180  # normalize

            turn_type = None

            if abs(angle) > 30:
                turn_type = "right" if angle > 0 else "left"

            if turn_type:

                edge_data = min(
                    self.graph.get_edge_data(path[i], path[i + 1]).values(),
                    key=lambda d: d.get("length", 1)
                )

                road_name = edge_data.get("name", "unknown road")

                if isinstance(road_name, list):
                    road_name = road_name[0]

                instructions.append({
                    "type": turn_type,
                    "road": road_name,
                    "distance_m": round(cumulative_distance, 1),
                    "index": i
                })

                cumulative_distance = 0  # reset for next instruction

        return instructions