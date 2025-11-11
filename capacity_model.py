import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Synthetic from projections: Years 2025-2050 scaled to daily-ish (interpolated)
np.random.seed(42)
days = 365 * 5  # 5 years for training
populations = np.linspace(1000000, 1078000, days) + np.random.normal(0, 10000, days)  # Trend + noise

# Normalize
mean = np.mean(populations)
std_dev = np.std(populations)
pop_norm = (populations - mean) / std_dev

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
X, y = create_sequences(pop_norm, seq_length)
X = torch.from_numpy(X).float().unsqueeze(2)
y = torch.from_numpy(y).float().unsqueeze(1)

# LSTM (same)
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

# Forecast 20 years (daily steps)
model.eval()
with torch.no_grad():
    last_seq = torch.from_numpy(pop_norm[-seq_length:]).float().unsqueeze(0).unsqueeze(2)
    forecasts = []
    for _ in range(365 * 20):
        pred = model(last_seq)
        forecasts.append(pred.item())
        last_seq = torch.cat((last_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(2)), dim=1)
forecasts = np.array(forecasts) * std_dev + mean

# Probabilistic factor from forecasts
num_simulations = 1000
simulated_factors = forecasts[-365*20:] / populations[0]  # End / start
updated_factor = np.percentile(simulated_factors, 90)

print(f"Updated probabilistic factor for capacity growth: {updated_factor:.2f}")
