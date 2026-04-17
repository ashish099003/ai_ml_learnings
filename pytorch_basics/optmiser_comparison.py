import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# 🎯 Comparing Optimizers

print("=" * 60)
print("OPTIMIZER COMPARISON")
print("=" * 60)

# Create a simple function to optimize (find minimum)
def create_test_model():
    return nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(100, 2)
y = (X[:, 0]**2 + X[:, 1]**2).unsqueeze(1) + torch.randn(100, 1) * 0.1 # y=x1**2​+x2**2​+noise

# Test different optimizers
optimizers_config = {
    'SGD': lambda p: optim.SGD(p, lr=0.01),
    'SGD with Momentum': lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
    'Adam': lambda p: optim.Adam(p, lr=0.01),
    'AdamW': lambda p: optim.AdamW(p, lr=0.01),
    'RMSprop': lambda p: optim.RMSprop(p, lr=0.01),
}

results = {}

for opt_name, opt_func in optimizers_config.items():
    model = create_test_model()
    optimizer = opt_func(model.parameters())
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(50):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    results[opt_name] = losses

# Plot convergence
plt.figure(figsize=(12, 6))
for opt_name, losses in results.items():
    plt.plot(losses, label=opt_name, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Convergence Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

# Learning Rate Scheduling
print("\nLEARNING RATE SCHEDULING")
print("-" * 40)

model = create_test_model()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Different schedulers
scheduler_options = {
    'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),
    'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50),
}

print("Scheduler effects on learning rate:")
for name, scheduler in scheduler_options.items():
    # Reset optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    if name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif name == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    lrs = []
    for epoch in range(50):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    print(f"{name}: Start LR={lrs[0]:.4f}, End LR={lrs[-1]:.4f}")