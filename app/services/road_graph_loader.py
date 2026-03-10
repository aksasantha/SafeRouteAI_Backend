import osmnx as ox
import networkx as nx

class RoadGraphLoader:
    def __init__(self, place_name: str, mode: str):

        assert mode in ["drive", "walk"], "mode must be 'drive' or 'walk'"

        print(f"🛣️ Loading {mode} enriched graph...")

        if mode == "drive":
            path = "app/data/kochi_drive_enriched.graphml"
        else:
            path = "app/data/kochi_walk_enriched.graphml"

        self.graph = ox.load_graphml(path)
        # Ensure numeric conversion of real features
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            for key in ["light_score", "crime_score",
                        "police_proximity_score", "crowd_score"]:
                if key in data:
                    data[key] = float(data[key])

        if not isinstance(self.graph, nx.MultiDiGraph):
            self.graph = nx.MultiDiGraph(self.graph)

        self._ensure_edge_lengths()
        self._convert_feature_types()
        self._build_edge_attr()

        print(
            f"✅ {mode.capitalize()} graph loaded: "
            f"{self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def _ensure_edge_lengths(self):
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            if "length" not in data:
                data["length"] = 1.0

    # 🔹 Step 2A: convert string → float
    def _convert_feature_types(self):
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            for key in [
                "light_score",
                "crime_score",
                "police_proximity_score",
                "crowd_score"
            ]:
                if key in data:
                    data[key] = float(data[key])

    # 🔹 Step 2B: rebuild edge_attr exactly as training expected
    def _build_edge_attr(self):
        for u, v, k, data in self.graph.edges(keys=True, data=True):

            # Convert from GraphML string → float safely
            light = float(data.get("light_score", 0.5))
            crime = float(data.get("crime_score", 0.5))
            police = float(data.get("police_proximity_score", 0.5))
            crowd = float(data.get("crowd_score", 0.5))

            data["edge_attr"] = [
                light,
                crime,
                police,
                crowd
            ]

    def get_graph(self):
        return self.graph