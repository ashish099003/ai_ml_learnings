import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import os

import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Train Shape: {x_train.shape}")



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(decoder_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)


            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# Dynamic early stopping monitor to protect system overhead
early_stop = keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

print("\n--- Initiating VAE Structural Training Loop ---")
vae.fit(
    x_train,
    epochs=30,
    batch_size=128,
    callbacks=[early_stop]
)


def plot_latent_space(vae, n=15, digit_size=28):
    """Plots a 2D grid of sampled digits from the latent space."""
    # Create a container image for the entire grid
    figure = np.zeros((digit_size * n, digit_size * n))

    # Linearly spaced coordinates corresponding to the 2D latent space
    # We sample between -2.0 and 2.0 standard deviations
    grid_x = np.linspace(-2.0, 2.0, n)
    grid_y = np.linspace(-2.0, 2.0, n)[::-1]  # Reverse y-axis to read top-to-bottom

    print("\n--- Generating 2D Latent Space Grid Mapping ---")

    for i, yi in enumerate(grid_y):
        for j, xj in enumerate(grid_x):
            # Create a coordinate point [x, y]
            z_sample = np.array([[xj, yi]])

            # Pass the coordinate point directly to the DECODER
            x_decoded = vae.decoder.predict(z_sample, verbose=0)

            # Reshape the output back to a 28x28 image slice
            digit = x_decoded[0].reshape(digit_size, digit_size)

            # Paste the slice into our master container image
            figure[
            i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size,
            ] = digit

    # Plot the final master container grid
    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)

    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)

    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("Latent Coordinate Z₁")
    plt.ylabel("Latent Coordinate Z₂")
    plt.imshow(figure, cmap="gray")
    plt.title("VAE 2D Latent Space Morphing Map")
    plt.show()





print("\n--- Generating Brand New Digits from Random Vectors ---")

custom_coordinates = np.array([
    [-1.5, 1.5],
    [1.5, 1.5],
    [0.0, 0.0],
    [-2.0, -2.0],
    [2.0, -2.0]
])

generated_digits = vae.decoder.predict(custom_coordinates)

plt.figure(figsize=(12, 3))
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    plt.imshow(generated_digits[i].reshape(28, 28), cmap="gray")
    plt.title(f"Coord: {custom_coordinates[i]}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plot_latent_space(vae)


# Create a directory to hold your model assets
os.makedirs("saved_vae_model", exist_ok=True)

vae.encoder.save("saved_vae_model/vae_encoder.keras")
vae.decoder.save("saved_vae_model/vae_decoder.keras")
print("\n[SUCCESS] Encoder and Decoder architectures saved successfully!")
