import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Synthetic loads: Hypothetical daily (e.g., wind + rain influence)
np.random.seed(42)
days = 365
loads = np.random.gamma(2.0, 1.0, days).cumsum()  # Skewed extremes

# Features: e.g., day, random weather proxy
features = np.column_stack((np.arange(days), np.random.normal(0, 1, days)))

# Normalize
mean_load = np.mean(loads)
std_load = np.std(loads)
loads_norm = (loads - mean_load) / std_load

# NN model (simple MLP as proxy for RF, since sklearn not available)
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

X = torch.from_numpy(features).float()
y = torch.from_numpy(loads_norm).float().unsqueeze(1)

model = NN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Predict future loads (e.g., next 365 with increased weather)
with torch.no_grad():
    future_features = np.column_stack((np.arange(days, days*2), np.random.normal(0, 1.5, days)))  # Increased std
    future_X = torch.from_numpy(future_features).float()
    forecasts = model(future_X).numpy().flatten() * std_load + mean_load

# Simulation
num_simulations = 5000
climate_uplift = 1.3
simulated_loads = np.random.gamma(2.0, np.mean(forecasts) / 2.0, num_simulations) * climate_uplift
updated_factor = np.percentile(simulated_loads, 99.9)

print(f"Updated safety factor for structural design: {updated_factor:.2f}")
