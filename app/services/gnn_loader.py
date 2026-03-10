import torch
from app.model.gnn_model import SafeRouteGNN

MODEL_PATH = "app/model/saferoute_gnn_v2.pth"

class GNNModelLoader:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = SafeRouteGNN(
            node_features=3,
            edge_features=4,
            hidden_dim=64
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device),
            strict=True)
        self.model.eval()
        
        print("Model parameters:", sum(p.numel() for p in self.model.parameters()))
        print("✅ SafeRouteGNN model loaded successfully")
        print("Loading model from:", MODEL_PATH)
        for name, param in self.model.named_parameters():
            print(name, param.mean().item())
            break
    
    @torch.no_grad()
    def predict(self, data):
        data = data.to(self.device)
        scores = self.model(
            data.x,
            data.edge_index,
            data.edge_attr
        )
        scores = scores.detach().cpu()

    # ✅ Force 1D output
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        return scores.numpy()


    def is_ready(self):
        return self.model is not None