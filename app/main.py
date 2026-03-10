from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.api.routes import router as routes_router
from app.api.route_test import router as route_test_router

from app.services.gnn_loader import GNNModelLoader
from app.services.road_graph_loader import RoadGraphLoader
from app.services.gnn_graph_injector import GNNGraphInjector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- STARTUP ----------
    print("🚀 SafeRouteAI backend starting...")

    # 1️⃣ Load trained GNN model
    gnn_loader = GNNModelLoader()
    app.state.gnn_model = gnn_loader

    # 2️⃣ Load raw road graphs
    drive_graph = RoadGraphLoader(
        "Kochi, Kerala, India", mode="drive"
    ).get_graph()

    walk_graph = RoadGraphLoader(
        "Kochi, Kerala, India", mode="walk"
    ).get_graph()

    # 3️⃣ Inject GNN safety scores into graphs
    injector = GNNGraphInjector(gnn_loader.model)

    print("🧠 Injecting GNN scores into DRIVE graph...")
    drive_graph = injector.inject(drive_graph)

    print("🧠 Injecting GNN scores into WALK graph...")
    walk_graph = injector.inject(walk_graph)

    # 4️⃣ Store AI-ready graphs
    app.state.road_graphs = {
        "drive": drive_graph,
        "walk": walk_graph
    }

    print("✅ Core services loaded successfully")

    yield

    # ---------- SHUTDOWN ----------
    print("🛑 SafeRouteAI backend shutting down...")


app = FastAPI(
    title="SafeRouteAI Backend",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for dev
    allow_credentials=True,
    allow_methods=["*"],  # IMPORTANT: allows OPTIONS
    allow_headers=["*"],
)

# Routers
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(routes_router)
app.include_router(route_test_router)


@app.get("/")
def root():
    return {"status": "SafeRouteAI backend running"}


@app.get("/model/status")
def model_status():
    return {
        "model": "SafeRouteGNN",
        "loaded": app.state.gnn_model.is_ready()
    }
