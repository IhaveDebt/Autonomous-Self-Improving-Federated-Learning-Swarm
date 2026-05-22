import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

# -------- Base Model --------
class ClientModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------- Federated Client --------
class Client:
    def __init__(self, client_id, data):
        self.client_id = client_id
        self.data = data
        self.model = None

    def local_train(self, global_model, epochs=3, lr=0.01):
        self.model = copy.deepcopy(global_model)

        opt = torch.optim.SGD(self.model.parameters(), lr=lr)

        for _ in range(epochs):
            x, y = self.data
            pred = self.model(x)
            loss = F.mse_loss(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        return self.model.state_dict(), loss.item()

# -------- Swarm Orchestrator --------
class FederatedSwarm:
    def __init__(self, num_clients, input_dim, hidden_dim, output_dim):
        self.global_model = ClientModel(input_dim, hidden_dim, output_dim)
        self.clients = [
            Client(i, self._generate_data(input_dim, output_dim))
            for i in range(num_clients)
        ]

    def _generate_data(self, input_dim, output_dim):
        x = torch.randn(32, input_dim)
        y = torch.randn(32, output_dim)
        return (x, y)

    def aggregate(self, updates):
        global_dict = self.global_model.state_dict()

        for k in global_dict.keys():
            global_dict[k] = torch.stack([u[k] for u in updates], dim=0).mean(dim=0)

        self.global_model.load_state_dict(global_dict)

    def inject_noise(self):
        with torch.no_grad():
            for p in self.global_model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

    def train_round(self):
        updates = []
        losses = []

        selected = random.sample(self.clients, k=len(self.clients)//2)

        for client in selected:
            update, loss = client.local_train(self.global_model)
            updates.append(update)
            losses.append(loss)

        self.aggregate(updates)

        if sum(losses) / len(losses) > 0.5:
            self.inject_noise()

        return sum(losses) / len(losses)

# -------- Run Simulation --------
swarm = FederatedSwarm(num_clients=10, input_dim=16, hidden_dim=32, output_dim=8)

for round in range(50):
    loss = swarm.train_round()
    print(f"[ROUND {round}] Loss: {loss:.4f}")
