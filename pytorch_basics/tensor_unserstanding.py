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


def create_tensor_arthimetic_operations():
    print("=" * 60)
    print("TENSOR CREATION METHODS")
    print("=" * 60)

    # 1. From Python lists
    tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
    print("From list:", tensor_from_list)

    # 2. From NumPy arrays
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_from_numpy = torch.from_numpy(numpy_array)
    print("\nFrom NumPy:\n", tensor_from_numpy)

    # 3. Random tensors (useful for initializing weights)
    random_tensor = torch.randn(3, 4)  # Normal distribution
    print("\nRandom (Normal):\n", random_tensor)

    # 4. Zeros and Ones (useful for initialization)
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    print("\nZeros:\n", zeros)
    print("\nOnes:\n", ones)

    # 5. Range tensors
    range_tensor = torch.arange(0, 10, 2)  # Start, stop, step
    print("\nRange:", range_tensor)

    # 6. Like tensors (same shape as another tensor)
    like_tensor = torch.zeros_like(random_tensor)
    print("\nZeros like random_tensor shape:", like_tensor.shape)

    print("=" * 60)
    print("TENSOR OPERATIONS")
    print("=" * 60)

    # Create sample tensors
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

    print("Tensor A:\n", a)
    print("\nTensor B:\n", b)

    # Arithmetic operations
    print("\n" + "=" * 40)
    print("ARITHMETIC OPERATIONS")
    print("=" * 40)

    print("Addition (a + b):\n", a + b)
    print("\nSubtraction (a - b):\n", a - b)
    print("\nMultiplication (a * b):\n", a * b)
    print("\nDivision (a / b):\n", a / b)

    # Matrix multiplication (crucial for neural networks!)
    print("\n" + "=" * 40)
    print("MATRIX MULTIPLICATION")
    print("=" * 40)

    print("Matrix multiplication (a @ b):\n", torch.matmul(a, b))
    print("Alternative (torch.mm):\n", torch.mm(a, b))

    # Statistical operations
    print("\n" + "=" * 40)
    print("STATISTICAL OPERATIONS")
    print("=" * 40)

    data = torch.randn(100)
    print(f"Mean: {data.mean():.4f}")
    print(f"Std: {data.std():.4f}")
    print(f"Min: {data.min():.4f}")
    print(f"Max: {data.max():.4f}")


def tensor_manipulations():
    # 🎯 Tensor Manipulation - Reshaping and Indexing

    print("=" * 60)
    print("TENSOR MANIPULATION")
    print("=" * 60)

    # Create a sample tensor
    x = torch.arange(12)
    print("Original tensor:", x)
    print("Shape:", x.shape)

    # Reshaping
    print("\n" + "=" * 40)
    print("RESHAPING")
    print("=" * 40)

    x_reshaped = x.reshape(3, 4)
    print("Reshaped to 3x4:\n", x_reshaped)

    x_reshaped2 = x.reshape(2, 6)
    print("\nReshaped to 2x6:\n", x_reshaped2)

    # View (similar to reshape but shares memory)
    x_view = x.view(4, 3)
    print("\nView as 4x3:\n", x_view)

    # Squeeze and Unsqueeze (remove/add dimensions)
    print("\n" + "=" * 40)
    print("SQUEEZE/UNSQUEEZE")
    print("=" * 40)

    y = torch.randn(1, 3, 1, 4)
    print("Original shape:", y.shape)
    y_squeezed = y.squeeze()
    print("After squeeze:", y_squeezed.shape)

    z = torch.randn(3, 4)
    z_unsqueezed = z.unsqueeze(0)  # Add dimension at position 0
    print("\nAfter unsqueeze(0):", z_unsqueezed.shape)

    # Indexing and Slicing
    print("\n" + "=" * 40)
    print("INDEXING & SLICING")
    print("=" * 40)

    tensor = torch.randn(3, 4)
    print("Original tensor:\n", tensor)
    print("\nFirst row:", tensor[0])
    print("Second column:", tensor[:, 1])
    print("Top-left 2x2:", tensor[:2, :2])


def autograd_basics():
    # 🎯 Understanding Autograd

    print("=" * 60)
    print("AUTOMATIC DIFFERENTIATION BASICS")
    print("=" * 60)

    # Create tensors with gradient tracking
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    print(f"x = {x}, requires_grad = {x.requires_grad}")
    print(f"y = {y}, requires_grad = {y.requires_grad}")

    # Perform operations
    z = x ** 2 + y ** 3  # z = x² + y³
    print(f"\nz = x² + y³ = {z}")

    # Compute gradients
    z.backward()  # Computes dz/dx and dz/dy

    print(f"\ndz/dx = 2x = {x.grad}")  # Should be 2*2 = 4
    print(f"dz/dy = 3y² = {y.grad}")  # Should be 3*3² = 27

    # 🎯 More Complex Example
    print("\n" + "=" * 60)
    print("COMPLEX GRADIENT COMPUTATION")
    print("=" * 60)

    # Reset gradients
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

    # Forward pass
    z = torch.sum(x * y)  # z = Σ(xi * yi)
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = sum(x * y) = {z}")

    # Backward pass
    z.backward()

    print(f"\nGradients:")
    print(f"dz/dx = y = {x.grad}")  # Gradient is y
    print(f"dz/dy = x = {y.grad}")  # Gradient is x


def gradient_tracking():
    # 🎯 Controlling Gradient Tracking

    print("=" * 60)
    print("GRADIENT TRACKING CONTROL")
    print("=" * 60)

    # Sometimes we don't want to track gradients (e.g., during inference)
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Method 1: torch.no_grad() context manager
    with torch.no_grad():
        y = x * 2
        print(f"Inside no_grad: y.requires_grad = {y.requires_grad}")

    # Method 2: .detach()
    y = x * 2
    y_detached = y.detach()
    print(f"\nDetached: y_detached.requires_grad = {y_detached.requires_grad}")

    # Method 3: Turn off globally (not recommended)
    torch.set_grad_enabled(False)
    z = x * 2
    print(f"\nGrad disabled: z.requires_grad = {z.requires_grad}")
    torch.set_grad_enabled(True)  # Turn back on

    # Real-world example: Computing loss but not updating certain layers
    print("\n" + "=" * 60)
    print("PRACTICAL EXAMPLE: FREEZING LAYERS")
    print("=" * 60)

    # Simulating a pre-trained layer (frozen) and a new layer (trainable)
    pretrained_weights = torch.randn(5, 3, requires_grad=False)  # Frozen
    new_weights = torch.randn(3, 2, requires_grad=True)  # Trainable

    input_data = torch.randn(10, 5)

    # Forward pass
    hidden = input_data @ pretrained_weights  # No gradients tracked
    output = hidden @ new_weights  # Gradients tracked

    loss = output.sum()
    loss.backward()

    print(f"Pretrained weights gradient: {pretrained_weights.grad}")  # None
    print(f"New weights gradient exists: {new_weights.grad is not None}")  # True


if __name__ == '__main__':
    # tensor_manipulations()
    # autograd_basics()
    gradient_tracking()