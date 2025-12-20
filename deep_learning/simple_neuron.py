import numpy as np
import pandas as pd

from test.deep_learning.touch_decision_neuron import TouchDecisionNeuron


class SimpleNeuron:

    def __init__(self, n_inputs=2, weights=None, bias=None):
        """
                Initialize the neuron with weights and bias.

                Args:
                    n_inputs: Number of input features
                    weights: Initial weights (if None, uses small random values)
                    bias: Initial bias (if None, starts at 0)
                """
        if weights is None:
            self.weights = np.random.rand(n_inputs) * 0.01
        else:
            self.weights = weights

        if bias is None:
            self.bias = 0.0
        else:
            self.bias = bias

        print(f"🔬 Neuron initialized!")
        print(f"   Weights: {self.weights}")
        print(f"   Bias: {self.bias}")

    def sigmoid(self,z):
        """
                Sigmoid activation function: squashes any value to [0, 1]

                Args:
                    z: Linear combination (can be any real number)

                Returns:
                    Value between 0 and 1 (probability-like)
                """
        return 1/(1 + np.exp(-z))

    def process(self, inputs):
        z = np.dot(self.weights,inputs) + self.bias
        output = self.sigmoid(z)
        return output, z

    def decide(self, inputs, threshold=0.5):
        """
                Make a binary decision based on inputs.

                Args:
                    inputs: Array of input values
                    threshold: Decision threshold (default 0.5)

                Returns:
                    Boolean decision and probability
        """
        probability, z = self.process(inputs)
        decision = probability >= threshold
        return decision, probability, z


if __name__=='__main__':
    print("\n" + "=" * 60)
    print("📧 EXAMPLE 2: Spam Email Detection Neuron")
    print("=" * 60)

    # Create a spam detection neuron
    spam_neuron = SimpleNeuron(n_inputs=3)
    spam_neuron.weights =np.array([2.0, -1.5, 3.0])  # [has_links, from_known, has_urgency]
    spam_neuron.bias = -1.0
    print("\n🔍 Spam Detection Features:")
    print("   1. Has suspicious links (0-1)")
    print("   2. From known sender (0-1)")
    print("   3. Has urgency words (0-1)")

    test_email = [("Suspicious Email", [0.9, 0.1, 0.8]), # Many Links, unknown sender, urgent
                  ("Normal email", [0.2, 0.9, 0.1]),  # Few links, known sender, not urgent
                  ("Borderline email", [0.5, 0.5, 0.6]) ]  # Some links, semi-known, somewhat urgent

    print("\n📊 Spam Detection Results:")
    for name, features in test_email:
        _, prob, z = spam_neuron.decide(features)
        is_spam = "🚫 SPAM" if prob > 0.5 else "✅ NOT SPAM"
        print(f"\n{name}:")
        print(f"   Features: {features}")
        print(f"   Linear combination: {z:.3f}")
        print(f"   Spam probability: {prob:.3f} ({prob * 100:.1f}%)")
        print(f"   Classification: {is_spam}")


