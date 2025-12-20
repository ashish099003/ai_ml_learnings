import numpy as np

def mini_batching():
    # ============================================
    # COMPLETE EXAMPLE: MINI BATCH
    # ============================================
    print("\n" + "=" * 60)
    print("🚀 COMPLETE EXAMPLE: Processing 3 Samples")
    print("=" * 60)

    # 3 samples, 3 classes each
    raw_scores = np.array([
        [2.0, 1.0, 0.1],  # Sample 1 scores
        [0.5, 2.5, 0.5],  # Sample 2 scores
        [1.0, 1.0, 1.0]  # Sample 3 scores
    ])
    classes = ['CAT', 'DOG', 'BIRD']
    print(f"   Classes: {classes}")

    true_labels = np.array([0, 1, 2])  # CAT, DOG, BIRD

    print(f"\n📥 Batch Input:")
    print(f"   3 samples, each with scores for 3 classes")
    print(f"   Raw scores:\n{raw_scores}")
    print(f"   True labels: {true_labels} (CAT, DOG, BIRD)")

    # Apply softmax to each sample
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stable version
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    probabilities = softmax(raw_scores)

    print(f"\n📊 After Softmax (probabilities):")
    for i in range(3):
        print(f"   Sample {i + 1}: [{probabilities[i, 0]:.3f}, {probabilities[i, 1]:.3f}, {probabilities[i, 2]:.3f}]")
        print(f"            Predicts: {classes[np.argmax(probabilities[i])]} (highest prob)")

    # Calculate cross-entropy loss for batch
    losses = -np.log(probabilities[range(3), true_labels])
    avg_loss = np.mean(losses)

    print(f"\n📉 Cross-Entropy Loss:")
    for i in range(3):
        correct_prob = probabilities[i, true_labels[i]]
        print(f"   Sample {i + 1}: -log({correct_prob:.3f}) = {losses[i]:.4f}")

    print(f"\n   Average Loss: {avg_loss:.4f}")


if __name__=='__main__':
    mini_batching()