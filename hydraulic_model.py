import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load data from CSV (skip header, extract "Average Flow (MGD)" column)
data = np.genfromtxt('all_data.csv', delimiter=',', skip_header=1, usecols=2, dtype=str)  # Column 2 is Average Flow
avg_flows = []
for val in data:
    try:
        flow = float(val.strip()) if val.strip() != 'N/A' else np.nan
        if not np.isnan(flow):
            avg_flows.append(flow)
    except ValueError:
        pass
avg_flows = np.array(avg_flows)
mean_avg = np.mean(avg_flows)
std_avg = np.std(avg_flows)

# Generate synthetic daily flow time-series (365 days, with trend)
np.random.seed(42)
days = 365
flows = np.cumsum(np.random.normal(mean_avg, std_avg, days))  # Cumulative for realistic trend

# Normalize
mean = np.mean(flows)
std_dev = np.std(flows)
flows_norm = (flows - mean) / std_dev

# Prepare sequences (lookback 30 days)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 30
X, y = create_sequences(flows_norm, seq_length)
X = torch.from_numpy(X).float().unsqueeze(2)  # (samples, seq_len, 1)
y = torch.from_numpy(y).float().unsqueeze(1)  # (samples, 1)

# LSTM model
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

# Train (simple loop)
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Forecast future (e.g., next 365 days)
model.eval()
with torch.no_grad():
    last_seq = torch.from_numpy(flows_norm[-seq_length:]).float().unsqueeze(0).unsqueeze(2)
    forecasts = []
    for _ in range(365):
        pred = model(last_seq)
        forecasts.append(pred.item())
        last_seq = torch.cat((last_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(2)), dim=1)
forecasts = np.array(forecasts) * std_dev + mean  # Denormalize

# Monte Carlo on forecasts
num_simulations = 5000
climate_uplift = 1.2
simulated_peaks = np.random.normal(np.mean(forecasts) * climate_uplift, np.std(forecasts), num_simulations)
updated_factor = np.percentile(simulated_peaks / mean_avg, 95)  # Ratio to average

print(f"Updated safety factor for hydraulic loading: {updated_factor:.2f}")
