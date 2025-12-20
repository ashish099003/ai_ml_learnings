import numpy as np

def softmax(z):
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
    return exp_z/np.sum(exp_z)

def softmax_using_maths():
    # ============================================
    # PART 2: SOFTMAX FUNCTION
    # ============================================
    print("\n" + "=" * 60)
    print("🌈 PART 2: SOFTMAX (Multi-class Classification)")
    print("=" * 60)

    print("\n🎯 Scenario: Classifying an image as CAT, DOG, or BIRD")

    # Raw scores from neural network (can be any values)
    z = np.array([2.0, 1.0, 0.1])
    classes = ['CAT', 'DOG', 'BIRD']

    print(f"\n📥 Input: Raw scores (logits) from neural network")
    print(f"   z = {z}")
    print(f"   Classes: {classes}")

    print(f"\n📝 Softmax Formula: P(class_i) = e^(z_i) / Σ(e^(z_j))")
    print(f"   Converts any values → probabilities that sum to 1")

    # Step 1: Exponentiate
    exp_z = np.exp(z)
    print(f"\n   Step 1: Exponentiate each score")
    print(f"          e^{z[0]} = e^2.0 = {exp_z[0]:.4f}")
    print(f"          e^{z[1]} = e^1.0 = {exp_z[1]:.4f}")
    print(f"          e^{z[2]} = e^0.1 = {exp_z[2]:.4f}")

    # Step 2: Sum
    sum_exp = np.sum(exp_z)
    print(f"\n   Step 2: Calculate sum")
    print(f"          Sum = {exp_z[0]:.4f} + {exp_z[1]:.4f} + {exp_z[2]:.4f}")
    print(f"              = {sum_exp:.4f}")

    # Step 3: Normalize
    softmax_output = exp_z / sum_exp
    print(f"\n   Step 3: Divide each by sum (normalize)")
    for i in range(3):
        print(f"          P({classes[i]}) = {exp_z[i]:.4f} / {sum_exp:.4f} = {softmax_output[i]:.4f}")

    print(f"\n📊 Final Probabilities:")
    for i in range(3):
        print(f"   {classes[i]}: {softmax_output[i]:.4f} ({softmax_output[i] * 100:.1f}%)")
    print(f"   Sum: {np.sum(softmax_output):.4f} ✅ (Always equals 1)")

    print(f"\n💡 Insight: Highest score (2.0) → Highest probability (65.9%)")
    print(f"           Softmax 'amplifies' differences between scores")
    return classes, softmax_output

if __name__=='__main__':
    # print(softmax([2.0,1.0,0.1]))
    print(softmax_using_maths())