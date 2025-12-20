import numpy as np

# Problem: Classify 3 types of flowers
# Single neuron can only do binary (yes/no)
# Solution: Use 3 neurons, one for each class!

class MultiCLassNetwork:
    def __init__(self,n_input=2, n_classes=3):

        self.w = np.random.randn(n_input, n_classes) * 0.01
        self.bias = np.zeros((1, n_classes))


    def forward(self, X):
        z = np.dot(X, self.w) + self.bias
        return self.softmax(z)

    def softmax(sef,z):
        """
        Converts raw scores to probabilities that sum to 1

        Example:
        Raw scores: [2.0, 1.0, 0.1]  (Class A looks best)

        Step 1: Exponentiate (make positive, enhance differences)
        e^2.0 = 7.39, e^1.0 = 2.72, e^0.1 = 1.11

        Step 2: Normalize (make them sum to 1)
        Total = 7.39 + 2.72 + 1.11 = 11.22

        Probabilities: [0.66, 0.24, 0.10]
        Interpretation: 66% Class A, 24% Class B, 10% Class C
        """
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)