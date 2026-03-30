import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)


def generate_fl_clients(num_samples=1000, heterogeneity="low_drift"):
    """
    Generate synthetic client datasets with varying degrees of statistical feature drift.

    All clients share the same underlying label-generation rule.
    Heterogeneity is introduced only through differences in input distributions.
    """
    drift_map = {
        "low_drift": 0.0,
        "medium_drift": 1.0,
        "high_drift": 3.0,
    }

    if heterogeneity not in drift_map:
        raise ValueError("heterogeneity must be 'low_drift', 'medium_drift', or 'high_drift'")

    drift = drift_map[heterogeneity]

    raw_data = {}
    all_features = []

    for i in range(1, 4):
        client_mean_shift = i * drift

        age = np.random.normal(40 + client_mean_shift, 10, num_samples)
        aum = np.random.normal(100000 + (client_mean_shift * 20000), 30000, num_samples)
        equity = np.random.normal(0.4 + (client_mean_shift * 0.05), 0.1, num_samples)

        x_raw = np.column_stack((age, aum, equity))
        raw_data[f"Client_{i}"] = x_raw
        all_features.append(x_raw)

    # Global scaling preserves relative drift between clients
    scaler = StandardScaler()
    scaler.fit(np.vstack(all_features))

    clients = {}

    for i in range(1, 4):
        x_scaled = scaler.transform(raw_data[f"Client_{i}"])

        # Shared ground-truth rule across all clients
        hidden_rule = (
            -1.5 * x_scaled[:, 0]
            + 0.5 * x_scaled[:, 1]
            + 2.0 * x_scaled[:, 2]
        )
        probabilities = 1 / (1 + np.exp(-hidden_rule))
        y = np.random.binomial(1, probabilities)

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        clients[f"Client_{i}"] = (x_tensor, y_tensor)

    return clients


class FLNeuralNetwork(nn.Module):
    """Small feed-forward neural network for binary classification."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)  # logits
        return x


def run_federated_simulation(heterogeneity_level, num_rounds=5, local_epochs=10):
    """
    Run iterative federated training and return client-level accuracies
    for the aggregated global model.
    """
    clients_data = generate_fl_clients(num_samples=500, heterogeneity=heterogeneity_level)
    global_model = FLNeuralNetwork()
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(num_rounds):
        global_weights = copy.deepcopy(global_model.state_dict())
        local_weights = []

        for _, (x, y) in clients_data.items():
            local_model = FLNeuralNetwork()
            local_model.load_state_dict(copy.deepcopy(global_weights))
            optimizer = optim.Adam(local_model.parameters(), lr=0.01)

            local_model.train()
            for _ in range(local_epochs):
                optimizer.zero_grad()
                logits = local_model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            local_weights.append(copy.deepcopy(local_model.state_dict()))

        # FedAvg (unweighted because all clients have identical sample sizes)
        averaged_weights = copy.deepcopy(global_weights)
        for key in averaged_weights.keys():
            averaged_weights[key] = torch.stack([w[key] for w in local_weights], dim=0).mean(dim=0)

        global_model.load_state_dict(averaged_weights)

    accuracies = []
    global_model.eval()

    with torch.no_grad():
        for _, (x, y) in clients_data.items():
            logits = global_model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == y).float().mean().item()
            accuracies.append(round(acc * 100, 2))

    return accuracies


def main():
    results_low = run_federated_simulation("low_drift")
    results_medium = run_federated_simulation("medium_drift")
    results_high = run_federated_simulation("high_drift")

    df = pd.DataFrame({
        "Heterogeneity": ["Low Drift", "Medium Drift", "High Drift"],
        "Client 1 Acc (%)": [results_low[0], results_medium[0], results_high[0]],
        "Client 2 Acc (%)": [results_low[1], results_medium[1], results_high[1]],
        "Client 3 Acc (%)": [results_low[2], results_medium[2], results_high[2]],
    })

    df["Mean Acc (%)"] = df.iloc[:, 1:4].mean(axis=1).round(2)
    df["Variance"] = df.iloc[:, 1:4].var(axis=1).round(2)

    print(df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.bar(df["Heterogeneity"], df["Variance"])
    plt.title("Impact of Client Data Drift on Variance in Federated Performance")
    plt.xlabel("Heterogeneity Level")
    plt.ylabel("Variance Across Client Accuracies")
    plt.tight_layout()
    plt.savefig("heterogeneity_variance.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
