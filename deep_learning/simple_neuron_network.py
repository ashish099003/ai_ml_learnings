import numpy as np
import matplotlib.pyplot as plt


class SimpleNeuralNetwork:
    """
    A simple neural network for multi-class classification.
    This is a complete, working implementation with extensive comments!
    """

    def __init__(self, n_features, n_classes, learning_rate=0.1):
        """
        Initialize the network.

        Args:
            n_features: Number of input features (e.g., 2 for x,y coordinates)
            n_classes: Number of output classes (e.g., 3 for A,B,C)
            learning_rate: How big our learning steps are (0.1 is usually good)
        """
        # Initialize weights with small random values
        # Why small? Large values can cause saturation in sigmoid/softmax
        self.W = np.random.randn(n_features, n_classes) * 0.01

        # Initialize biases to zero
        # Starting neutral - no initial preference for any class
        self.b = np.zeros((1, n_classes))

        # Learning rate - how much we adjust weights each step
        self.lr = learning_rate

        # Store dimensions for later use
        self.n_features = n_features
        self.n_classes = n_classes

        # History for plotting
        self.loss_history = []

        print(f"🧠 Neural Network initialized!")
        print(f"   Input features: {n_features}")
        print(f"   Output classes: {n_classes}")
        print(f"   Total parameters: {n_features * n_classes + n_classes}")

    def softmax(self, Z):
        """
        Convert raw scores to probabilities.

        Args:
            Z: Raw scores from linear combination, shape (n_samples, n_classes)

        Returns:
            Probabilities that sum to 1 for each sample
        """
        # Subtract max for numerical stability (prevent overflow)
        # This doesn't change the result, just makes computation safer

        Z_stable = Z - np.max(Z, axis=1, keepdims=True)

        # Exponentiate and normalize
        exp_Z = np.exp(Z_stable)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        """
        Pass data through the network to get predictions.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            Predicted probabilities for each class
        """
        # Linear combination: Z = XW + b
        # This is like weighted voting - each feature votes for each class
        self.Z = np.dot(X, self.W) + self.b

        # Apply softmax to get probabilities
        # Converts any real numbers to probabilities between 0 and 1
        self.A = self.softmax(self.Z)

        return self.A

    def compute_loss(self, y_true):
        """
        Calculate how wrong our predictions are.

        Args:
            y_true: True class labels (integers like 0, 1, 2)

        Returns:
            Cross-entropy loss (lower is better)
        """
        m = len(y_true)

        # Get the predicted probability for the correct class of each sample
        # If sample i belongs to class j, we want A[i,j] to be close to 1
        correct_class_probs = self.A[range(m), y_true]

        # Cross-entropy: -log(probability of correct class)
        # If prob = 1 (perfect), loss = 0
        # If prob = 0.1 (bad), loss = 2.3
        # Add small epsilon to avoid log(0)
        loss = -np.mean(np.log(correct_class_probs + 1e-7))

        return loss

    def backward_propagation(self, X, y_true):
        """
        Calculate gradients - how to adjust weights to reduce error.

        This is where the magic happens!
        We figure out which weights caused the error and by how much.

        Args:
            X: Input data
            y_true: True labels
        """
        m = len(y_true)

        # Step 1: Calculate error at output layer
        # For softmax + cross-entropy, this has a beautiful simple form:
        # gradient = predictions - truth
        self.dZ = self.A.copy()
        self.dZ[range(m), y_true] -= 1
        self.dZ = self.dZ / m  # Average over all samples

        # Step 2: Calculate weight gradients
        # How much did each weight contribute to the error?
        # dW = X^T @ dZ (input × error gives weight gradient)
        self.dW = np.dot(X.T, self.dZ)

        # Step 3: Calculate bias gradients
        # Bias gradient is just the sum of errors
        self.db = np.sum(self.dZ, axis=0, keepdims=True)

    def update_parameters(self):
        """
        Adjust weights and biases based on gradients.
        This is gradient descent - taking a step downhill toward lower loss.
        """
        # Take a step in the opposite direction of gradient
        # (Gradient points uphill, we want to go downhill)
        self.W = self.W - self.lr * self.dW
        self.b = self.b - self.lr * self.db

    def train_step(self, X, y):
        """
        One complete training step.

        Args:
            X: Training data
            y: Training labels

        Returns:
            Loss for this step
        """
        # 1. Forward pass - make predictions
        self.forward_propagation(X)

        # 2. Calculate loss - how wrong were we?
        loss = self.compute_loss(y)

        # 3. Backward pass - calculate gradients
        self.backward_propagation(X, y)

        # 4. Update - adjust weights
        self.update_parameters()

        return loss

    def fit(self, X, y, epochs=1000, verbose=True):
        """
        Train the network on data.

        Args:
            X: Training data
            y: Training labels
            epochs: Number of training iterations
            verbose: Whether to print progress
        """
        print(f"\n🚀 Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # One training step
            loss = self.train_step(X, y)
            self.loss_history.append(loss)

            # Print progress
            if verbose and epoch % 100 == 0:
                accuracy = self.accuracy(X, y)
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

        print(f"✅ Training complete!")

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input data

        Returns:
            Predicted class labels
        """
        # Get probabilities
        probabilities = self.forward_propagation(X)

        # Return class with highest probability
        return np.argmax(probabilities, axis=1)

    def accuracy(self, X, y):
        """Calculate prediction accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def plot_loss(self):
        """Visualize training progress."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.grid(True, alpha=0.3)
        plt.show()

def plot_decision_boundary(nn, X, y):
    """
    Visualize what our network learned!
    Shows the decision regions for each class.
    """
    # Set up the plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict for every point in the mesh
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot 1: Decision regions
    axes[0].contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral,
                    edgecolors='black', s=50)
    axes[0].set_title('Decision Boundaries')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # Plot 2: Probability heatmap for class 0
    probabilities = nn.forward_propagation(np.c_[xx.ravel(), yy.ravel()])
    Z_prob = probabilities[:, 0].reshape(xx.shape)

    im = axes[1].contourf(xx, yy, Z_prob, levels=20, cmap='RdYlBu_r', alpha=0.8)
    axes[1].scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o',
                    edgecolors='black', s=50, label='Class 0')
    axes[1].scatter(X[y==1, 0], X[y==1, 1], c='yellow', marker='s',
                    edgecolors='black', s=50, label='Class 1')
    axes[1].scatter(X[y==2, 0], X[y==2, 1], c='blue', marker='^',
                    edgecolors='black', s=50, label='Class 2')
    axes[1].set_title('Probability Heatmap for Class 0')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()

    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.show()

# Visualize what our network learned

# Create the famous spiral dataset
def create_spiral_dataset(n_points=100, n_classes=3):
    """
    Create a spiral dataset for testing our neural network.
    This is a challenging dataset that requires non-linear boundaries.
    """
    X = []
    y = []

    for class_idx in range(n_classes):
        # Create points along a spiral
        theta = np.linspace(class_idx * 4, (class_idx + 1) * 4, n_points)
        radius = np.linspace(0.1, 1, n_points)

        # Add some noise for realism
        theta += np.random.randn(n_points) * 0.2

        # Convert to x,y coordinates
        x1 = radius * np.sin(theta)
        x2 = radius * np.cos(theta)

        X.extend(np.column_stack([x1, x2]))
        y.extend([class_idx] * n_points)

    return np.array(X), np.array(y)

# Generate dataset
X_train, y_train = create_spiral_dataset(n_points=100, n_classes=3)

# Create and train network
nn = SimpleNeuralNetwork(n_features=2, n_classes=3, learning_rate=1.0)
nn.fit(X_train, y_train, epochs=500, verbose=True)

# Visualize results
nn.plot_loss()

# Check final accuracy
final_accuracy = nn.accuracy(X_train, y_train)
print(f"\n🎯 Final Training Accuracy: {final_accuracy:.2%}")



if __name__=='__main__':
    # create_spiral_dataset(n_points=100, n_classes=3)
    plot_decision_boundary(nn, X_train, y_train)
