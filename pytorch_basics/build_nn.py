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



class SimpleNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet,self).__init__()

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)


    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class AdvancedNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.2):
        """
        Advanced network with multiple hidden layers, dropout, and batch norm

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes
            dropout_prob: Dropout probability for regularization
        """
        super(AdvancedNet, self).__init__()

        # Build layers dynamically
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            # Add linear layer
            layers.append(nn.Linear(prev_size, hidden_size))

            # Add batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))

            # Add activation
            layers.append(nn.ReLU())

            # Add dropout for regularization (except last layer)
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_prob))

            prev_size = hidden_size

        # Create sequential container
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        # Pass through hidden layers
        x = self.hidden_layers(x)

        # Output layer (no activation for flexibility)
        x = self.output_layer(x)

        return x


def create_simple_model():
    # Create an instance of the network
    # 🎯 Your First Neural Network

    print("=" * 60)
    print("BUILDING A SIMPLE NEURAL NETWORK")
    print("=" * 60)
    model = SimpleNet(input_size=10, hidden_size=20, output_size=2)

    print("Model Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # View individual layers
    print("\nLayer Details:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
def create_advance_model():
    # Create an advanced model
    # 🎯 Building a More Complex Network

    print("=" * 60)
    print("ADVANCED NEURAL NETWORK WITH MULTIPLE FEATURES")
    print("=" * 60)

    model_advanced = AdvancedNet(
        input_size=20,
        hidden_sizes=[64, 32, 16],
        output_size=3,
        dropout_prob=0.3
    )

    print("Advanced Model Architecture:")
    print(model_advanced)

    # Test with random input
    test_input = torch.randn(5, 20)  # Batch of 5 samples, 20 features each
    output = model_advanced(test_input)

    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Visualize parameter count by layer
    print("\nParameter Count by Layer:")
    for name, module in model_advanced.named_modules():
        if isinstance(module, nn.Linear):
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {params:,} parameters")

if __name__=='__main__':
    # create_simple_model()
    create_advance_model()