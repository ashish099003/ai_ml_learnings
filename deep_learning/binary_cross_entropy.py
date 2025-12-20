import numpy as np

from test.deep_learning.softmax import softmax_using_maths


def cross_entropy_loss(y_true, y_pred):

    one_hot = np.zeros_like(y_pred)
    one_hot[y_true] = 1

    return -np.sum(one_hot*np.log(y_pred))


def binary_cross_entropy_using_simple_maths():
    print("=" * 60)
    print("🧮 NEURAL NETWORK MATH - SIMPLE EXAMPLES WITH NUMBERS")
    print("=" * 60)

    # ============================================
    # PART 1: BINARY CROSS-ENTROPY (LOG LOSS)
    # ============================================
    print("\n" + "=" * 60)
    print("📊 PART 1: BINARY CROSS-ENTROPY (Binary Classification)")
    print("=" * 60)

    print("\n🎯 Scenario: Classifying if an email is SPAM (1) or NOT SPAM (0)")
    # Example 1: Good prediction
    y_true_1 = 1  # Actual: This IS spam
    y_pred_1 = 0.9  # Model says: 90% chance it's spam

    print(f"\n✅ Example 1 - Good Prediction:")
    print(f"   Truth: {y_true_1} (It IS spam)")
    print(f"   Prediction: {y_pred_1} (90% sure it's spam)")

    # Binary Cross-Entropy Formula
    print(f"\n📝 Formula: Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]")
    print(f"   Where: y = true label, ŷ = predicted probability")

    # Calculate step by step
    term1 = y_true_1 * np.log(y_pred_1)
    term2 = (1 - y_true_1) * np.log(1 - y_pred_1)
    loss_1 = -(term1 + term2)

    print(f"\n   Step 1: First term = {y_true_1} * log({y_pred_1})")
    print(f"          = {y_true_1} * {np.log(y_pred_1):.4f}")
    print(f"          = {term1:.4f}")

    print(f"\n   Step 2: Second term = (1-{y_true_1}) * log(1-{y_pred_1})")
    print(f"          = {1 - y_true_1} * log({1 - y_pred_1})")
    print(f"          = {1 - y_true_1} * {np.log(1 - y_pred_1):.4f}")
    print(f"          = {term2:.4f}")

    print(f"\n   Step 3: Loss = -({term1:.4f} + {term2:.4f})")
    print(f"          = -{term1:.4f}")
    print(f"          = {loss_1:.4f} ✅ (Small loss - good!)")

    # Example 2: Bad prediction
    y_true_2 = 1  # Actual: This IS spam
    y_pred_2 = 0.1  # Model says: Only 10% chance it's spam (WRONG!)

    print(f"\n❌ Example 2 - Bad Prediction:")
    print(f"   Truth: {y_true_2} (It IS spam)")
    print(f"   Prediction: {y_pred_2} (Only 10% sure it's spam)")

    term1 = y_true_2 * np.log(y_pred_2)
    term2 = (1 - y_true_2) * np.log(1 - y_pred_2)
    loss_2 = -(term1 + term2)

    print(f"\n   Loss = -[{y_true_2}*log({y_pred_2}) + {1 - y_true_2}*log({1 - y_pred_2})]")
    print(f"        = -[{term1:.4f} + {term2:.4f}]")
    print(f"        = {loss_2:.4f} ❌ (Large loss - bad!)")

    print(f"\n💡 Insight: Good prediction → Small loss ({loss_1:.4f})")
    print(f"           Bad prediction → Large loss ({loss_2:.4f})")

def cross_entropy_loss_multi_class(classes, softmax_output):
    # ============================================
    # PART 3: CROSS-ENTROPY LOSS (Multi-class)
    # ============================================
    print("\n" + "=" * 60)
    print("🎲 PART 3: CROSS-ENTROPY LOSS (Multi-class)")
    print("=" * 60)

    print("\n🎯 Scenario: Our model predicted probabilities for CAT, DOG, BIRD")

    # Predictions from softmax
    predictions = softmax_output  # [0.659, 0.243, 0.099]
    true_class = 0  # Actual class: CAT (index 0)

    print(f"\n📥 Input:")
    print(f"   Predictions: CAT={predictions[0]:.3f}, DOG={predictions[1]:.3f}, BIRD={predictions[2]:.3f}")
    print(f"   True class: {classes[true_class]} (index {true_class})")

    print(f"\n📝 Cross-Entropy Formula: Loss = -log(P_correct_class)")
    print(f"   Only look at probability of the TRUE class")

    # Calculate loss
    loss = -np.log(predictions[true_class])

    print(f"\n   Calculation:")
    print(f"   Loss = -log(P_{classes[true_class]})")
    print(f"        = -log({predictions[true_class]:.4f})")
    print(f"        = -{np.log(predictions[true_class]):.4f}")
    print(f"        = {loss:.4f} ✅ (Good - model was 66% confident)")

    # Compare with wrong prediction
    print(f"\n🔄 What if the image was actually a BIRD?")
    true_class_wrong = 2  # BIRD
    loss_wrong = -np.log(predictions[true_class_wrong])

    print(f"   Loss = -log(P_BIRD)")
    print(f"        = -log({predictions[true_class_wrong]:.4f})")
    print(f"        = {loss_wrong:.4f} ❌ (Bad - model only gave 10% to BIRD)")

if __name__=='__main__':
    # binary_cross_entropy_using_simple_maths()
    # print(cross_entropy_loss([0,1,2], [0.2, 0.7, 0.1]))
    classes, softmax_output = softmax_using_maths()
    cross_entropy_loss_multi_class(classes,softmax_output)