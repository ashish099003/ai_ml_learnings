from test.deep_learning.simple_neuron import SimpleNeuron
import numpy as np
from test.deep_learning.touch_decision_neuron import TouchDecisionNeuron


def step_by_step():
    # ============================================
    # EXAMPLE 3: Step-by-Step Calculation
    # ============================================
    print("\n" + "=" * 60)
    print("📝 EXAMPLE 3: Manual Step-by-Step Calculation")
    print("=" * 60)

    demo_neuron = SimpleNeuron(weights=[0.5, 0.3], n_inputs=2, bias=0.2)
    # Input values
    x1, x2 = 2.0, 3.0
    print(f"\n📥 Inputs: x1={x1}, x2={x2}")
    print(f"⚙️ Weights: w1={demo_neuron.weights[0]}, w2={demo_neuron.weights[1]}")
    print(f"📍 Bias: b={demo_neuron.bias}")

    # Manual calculation
    print("\n🔢 Step-by-Step Calculation:")
    print(f"   Step 1: Calculate weighted sum")
    print(f"          z = w1*x1 + w2*x2 + b")
    print(f"          z = {demo_neuron.weights[0]}*{x1} + {demo_neuron.weights[1]}*{x2} + {demo_neuron.bias}")
    print(f"          z = {demo_neuron.weights[0] * x1} + {demo_neuron.weights[1] * x2} + {demo_neuron.bias}")
    z_manual = demo_neuron.weights[0] * x1 + demo_neuron.weights[1] * x2 + demo_neuron.bias
    print(f"          z = {z_manual}")

    print(f"\n   Step 2: Apply sigmoid activation")
    print(f"          output = 1 / (1 + e^(-z))")
    print(f"          output = 1 / (1 + e^(-{z_manual}))")
    print(f"          output = 1 / (1 + {np.exp(-z_manual):.4f})")
    output_manual = 1 / (1 + np.exp(-z_manual))
    print(f"          output = {output_manual:.4f}")

    # Verify with neuron's process method
    output_auto, z_auto = demo_neuron.process([x1, x2])
    print(f"\n✅ Verification using neuron.process():")
    print(f"   Output: {output_auto:.4f}")
    print(f"   Match: {np.isclose(output_manual, output_auto)}")

    # ============================================
    # VISUALIZATION
    # ============================================
    print("\n" + "=" * 60)
    print("📊 BONUS: Visualizing Neuron Behavior")
    print("=" * 60)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplot(1, 3, figsize=(15, 4))

    # Plot 1: Linear combination (z values)
    z_values = np.linspace(-5, 5, 100)
    axes[0].plot(z_values, z_values, 'b-', linewidth=2)
    axes[0].set_title('Step 1: Linear Combination')
    axes[0].set_xlabel('Weighted Sum (z)')
    axes[0].set_ylabel('z')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Plot 2: Sigmoid activation
    sigmoid_values = 1 / (1 + np.exp(-z_values))
    axes[1].plot(z_values, sigmoid_values, 'r-', linewidth=2)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision threshold')
    axes[1].set_title('Step 2: Sigmoid Activation')
    axes[1].set_xlabel('z')
    axes[1].set_ylabel('Sigmoid(z)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Decision regions for touch neuron
    temp_range = np.linspace(0, 1, 50)
    fam_range = np.linspace(0, 1, 50)
    temp_grid, fam_grid = np.meshgrid(temp_range, fam_range)

    touch_neuron_viz = TouchDecisionNeuron()
    decisions = np.zeros_like(temp_grid)

    for i in range(len(temp_range)):
        for j in range(len(fam_range)):
            _, prob, _ = touch_neuron_viz.decide([temp_grid[i, j], fam_grid[i, j]])
            decisions[i, j] = prob

    contour = axes[2].contourf(temp_grid, fam_grid, decisions, levels=20, cmap='RdYlGn_r')
    axes[2].contour(temp_grid, fam_grid, decisions, levels=[0.5], colors='black', linewidths=2)
    axes[2].set_title('Touch Decision Regions')
    axes[2].set_xlabel('Temperature (0=cold, 1=hot)')
    axes[2].set_ylabel('Familiarity (0=unknown, 1=familiar)')
    plt.colorbar(contour, ax=axes[2], label='Touch Probability')

    # Add example points
    examples = [
        (0.9, 0.2, 'Hot Unknown'),
        (0.3, 0.9, 'Cool Familiar'),
        (0.6, 0.7, 'Warm Semi-familiar')
    ]
    for temp, fam, label in examples:
        axes[2].plot(temp, fam, 'ko', markersize=8)
        axes[2].annotate(label, (temp, fam), xytext=(5, 5),
                         textcoords='offset points', fontsize=8)

    plt.tight_layout()
    plt.show()

    print("\n✅ Complete! The neuron is the building block of neural networks.")
    print("   Multiple neurons together = Neural Network! 🧠")


if __name__=='__main__':
    step_by_step()