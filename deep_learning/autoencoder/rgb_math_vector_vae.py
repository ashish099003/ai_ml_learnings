import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD THE DATA
print("--- Loading Dataset for Vector Math Operations ---")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
y_test = y_test.flatten()

# CIFAR-10 Class Map reference labels:
# 1 = Automobile, 2 = Bird, 3 = Cat, 4 = Deer, 5 = Dog, 7 = Horse, etc.

# 2. RELOAD YOUR TRAINED ENCODER AND DECODER
# (Make sure you saved them from the previous script run!)
try:
    # If using custom layers like our 'Sampling' layer, specify custom_objects
    from rgb_vae import Sampling  # Import the Sampling layer class from your previous file

    encoder = keras.models.load_model("saved_vae_model/vae_encoder.keras", custom_objects={"Sampling": Sampling})
    decoder = keras.models.load_model("saved_vae_model/vae_decoder.keras")
    print("[SUCCESS] Loaded trained models from disk.")
except:
    print("[FALLBACK] Saved model files not found. Please make sure to run the training script ")
    print("and call vae.encoder.save() and vae.decoder.save() first!")
    exit()

# 3. CHOOSE TARGET IMAGES FOR THE MATH OPERATION
# Let's say we want to morph the concept of a "Horse" into an "Automobile" (Mechanical Feature Morphing)
# Or find two specific items to swap styles.

# Find indexes of some horses and cars in our test set
horse_indices = np.where(y_test == 7)[0]
car_indices = np.where(y_test == 1)[0]

# Pick representative source images
img_horse = x_test[horse_indices[0]]
img_car = x_test[car_indices[0]]

# 4. EXTRACT THEIR LATENT VECTORS (ENCODE)
# The encoder returns [z_mean, z_log_var, z]. We want the actual sample coordinate 'z'.
_, _, z_horse = encoder.predict(np.expand_dims(img_horse, 0), verbose=0)
_, _, z_car = encoder.predict(np.expand_dims(img_car, 0), verbose=0)

# 5. PERFORM VECTOR MATHEMATICS (THE INTERPOLATION / MORPH)
# Instead of a strict binary addition, we can step smoothly along the line
# connecting the Horse vector to the Car vector to see the feature evolution!
print("--- Performing Latent Vector Linear Math Steps ---")

steps = 5
interpolated_vectors = []
for alpha in np.linspace(0.0, 1.0, steps):
    # Linear interpolation formula: z_new = (1 - alpha) * z_A + alpha * z_B
    # alpha = 0.0 means 100% Horse
    # alpha = 1.0 means 100% Car
    z_mixed = (1 - alpha) * z_horse + alpha * z_car
    interpolated_vectors.append(z_mixed)

# Concatenate all generated vectors into a single batch for fast prediction
interpolated_vectors = np.vstack(interpolated_vectors)

# 6. DECODE THE NEW MATH VECTORS BACK TO RGB IMAGES
generated_images = decoder.predict(interpolated_vectors, verbose=0)

# 7. PLOT THE TRANSFORMATION TRAJECTORY
plt.figure(figsize=(15, 4))
for i in range(steps):
    ax = plt.subplot(1, steps, i + 1)
    plt.imshow(generated_images[i])

    # Calculate percentage layout titles
    pct_car = int(np.linspace(0, 100, steps)[i])
    pct_horse = 100 - pct_car
    plt.title(f"{pct_horse}% Horse\n{pct_car}% Car", fontsize=10)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.suptitle("Latent Vector Space Algebra Simulation Trajectory", fontsize=14, weight='bold')
plt.show()
