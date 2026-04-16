import torch
import torch.nn as nn
import matplotlib.pyplot as plt


print("=" * 60)
print("ACTIVATION FUNCTIONS COMPARISON")
print("=" * 60)

# Create input range
x = torch.linspace(-5, 5, 100)

# Define activation functions
activations = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'LeakyReLU': nn.LeakyReLU(0.1),
    'ELU': nn.ELU(),
    'GELU': nn.GELU()
}

# Plot all activation functions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (name, activation) in enumerate(activations.items()):
    with torch.no_grad():
        y = activation(x)

    axes[idx].plot(x.numpy(), y.numpy(), linewidth=2)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_title(f'{name} Activation', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Input')
    axes[idx].set_ylabel('Output')
    axes[idx].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[idx].axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# Performance comparison
print("\nActivation Function Properties:")
print("-" * 50)
print(f"{'Function':<12} {'Range':<20} {'Best Use Case'}")
print("-" * 50)
print(f"{'ReLU':<12} {'[0, ∞)':<20} {'Hidden layers (default)'}")
print(f"{'Sigmoid':<12} {'(0, 1)':<20} {'Binary classification'}")
print(f"{'Tanh':<12} {'(-1, 1)':<20} {'Centered data'}")
print(f"{'LeakyReLU':<12} {'(-∞, ∞)':<20} {'Avoiding dead neurons'}")
print(f"{'ELU':<12} {'(-α, ∞)':<20} {'Faster convergence'}")
print(f"{'GELU':<12} {'Smooth':<20} {'Transformers, BERT'}")