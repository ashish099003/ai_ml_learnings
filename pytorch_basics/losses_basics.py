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

# 🎯 Understanding Loss Functions

print("=" * 60)
print("LOSS FUNCTIONS DEMONSTRATION")
print("=" * 60)

# Set up sample data
torch.manual_seed(42)
batch_size = 10
num_classes = 3

# Regression example (MSE Loss)
print("REGRESSION - MSE Loss")
print("-" * 40)

# True values and predictions for house prices (scaled)
y_true_reg = torch.randn(batch_size, 1) * 100 + 200  # True prices
y_pred_reg = y_true_reg + torch.randn(batch_size, 1) * 20  # Add noise

mse_loss = nn.MSELoss()
loss_value = mse_loss(y_pred_reg, y_true_reg)

print(f"True values (first 3): {y_true_reg[:3].squeeze().numpy()}")
print(f"Predictions (first 3): {y_pred_reg[:3].squeeze().numpy()}")
print(f"MSE Loss: {loss_value.item():.4f}")

# Classification example (Cross-Entropy Loss)
print("\nCLASSIFICATION - Cross-Entropy Loss")
print("-" * 40)

# Random logits (raw predictions) and true labels
logits = torch.randn(batch_size, num_classes)
y_true_class = torch.randint(0, num_classes, (batch_size,))

ce_loss = nn.CrossEntropyLoss()
loss_value_ce = ce_loss(logits, y_true_class)

print(f"Logits shape: {logits.shape}")
print(f"True labels (first 5): {y_true_class[:5].numpy()}")
print(f"Cross-Entropy Loss: {loss_value_ce.item():.4f}")

# Binary Classification (BCE Loss)
print("\nBINARY CLASSIFICATION - BCE Loss")
print("-" * 40)

# Predictions (after sigmoid) and true binary labels
y_pred_binary = torch.sigmoid(torch.randn(batch_size, 1))
y_true_binary = torch.randint(0, 2, (batch_size, 1)).float()

bce_loss = nn.BCELoss()
loss_value_bce = bce_loss(y_pred_binary, y_true_binary)

print(f"Predictions (first 5): {y_pred_binary[:5].squeeze().numpy()}")
print(f"True labels (first 5): {y_true_binary[:5].squeeze().numpy()}")
print(f"BCE Loss: {loss_value_bce.item():.4f}")

# Custom Loss Function
print("\nCUSTOM LOSS FUNCTION")
print("-" * 40)

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

focal_loss = FocalLoss()
loss_value_focal = focal_loss(logits, y_true_class)
print(f"Focal Loss: {loss_value_focal.item():.4f}")