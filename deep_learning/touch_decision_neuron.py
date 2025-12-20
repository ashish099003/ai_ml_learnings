import numpy as np

from test.deep_learning.simple_neuron import SimpleNeuron


class TouchDecisionNeuron(SimpleNeuron):

    def __init__(self):
        weights = [-2.0, 1.5]  # [temperature_weight, familiarity_weight]
        bias = 0.5  # Slightly cautious
        super().__init__(n_inputs=2, weights=weights, bias=bias)
        print("\n🧠 Touch Decision Neuron Configuration:")
        print("   Temperature weight: -2.0 (negative = avoid hot)")
        print("   Familiarity weight: 1.5 (positive = trust familiar)")
        print("   Bias: 0.5 (slightly cautious)")

    def should_i_touch(self, temperature, familiarity):

        inputs = np.array([temperature, familiarity])
        decision, probability, z = self.decide(inputs)

        print(f"\n📊 Analysis:")
        print(f"   Inputs: Temperature={temperature}, Familiarity={familiarity}")
        print(f"   Linear combination (z): {z:.3f}")
        print(f"   Probability of touching: {probability:.3f} ({probability * 100:.1f}%)")

        if decision:
            return "✅ Safe to touch!", probability
        else:
            return "⚠️ Don't touch!", probability



if __name__=='__main__':

    touch_neuron = TouchDecisionNeuron()
    # Test Case 1: Hot and Unknown
    # print("\n" + "-" * 40)
    # print("Test 1: Hot Iron (hot & unknown)")
    # result, prob = touch_neuron.should_i_touch(temperature=0.9, familiarity=0.2)
    # print(f"Decision: {result}")
    #
    # # Test Case 2: Cold and Familiar
    # print("\n" + "-" * 40)
    # print("Test 2: Your Phone (cool & familiar)")
    # result, prob = touch_neuron.should_i_touch(temperature=0.3, familiarity=0.9)
    # print(f"Decision: {result}")
    #
    # # Test Case 3: Warm and Semi-familiar
    # print("\n" + "-" * 40)
    # print("Test 3: Coffee Mug (warm & semi-familiar)")
    # result, prob = touch_neuron.should_i_touch(temperature=0.6, familiarity=0.7)
    # print(f"Decision: {result}")
