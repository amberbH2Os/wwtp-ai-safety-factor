import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Updated with BOD stats: Assume influent mean=300 mg/L (typical), std=100; from table limits
np.random.seed(42)
days = 365
bod_values = np.cumsum(np.random.normal(300, 100, days))  # Synthetic series

# Normalize
mean = np.mean(bod_values)
std_dev = np.std(bod_values)
bod_norm = (bod_values - mean) / std_dev

# Sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 30
X, y = create_sequences(bod_norm, seq_length)
X = torch.from_numpy(X).float().unsqueeze(2)
y = torch.from_numpy(y).float().unsqueeze(1)

# LSTM (same as above)
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

model = LSTM()
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

# Forecast
model.eval()
with torch.no_grad():
    last_seq = torch.from_numpy(bod_norm[-seq_length:]).float().unsqueeze(0).unsqueeze(2)
    forecasts = []
    for _ in range(365):
        pred = model(last_seq)
        forecasts.append(pred.item())
        last_seq = torch.cat((last_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(2)), dim=1)
forecasts = np.array(forecasts) * std_dev + mean

# Simulation with removal (≥85%) and limit (<30 mg/L)
num_simulations = 10000
upset_probability = 0.1
removal_min = 0.85
base_loads = np.random.normal(np.mean(forecasts), np.std(forecasts), num_simulations)
upsets = np.random.choice([1.0, 1.5], num_simulations, p=[1 - upset_probability, upset_probability])
simulated_loads = base_loads * upsets
simulated_effluent = simulated_loads * (1 - removal_min)
updated_factor = np.percentile(simulated_loads[simulated_effluent < 30], 99) / 300  # Normalized to typical influent

print(f"Updated safety factor for biological loading: {updated_factor:.2f}")
