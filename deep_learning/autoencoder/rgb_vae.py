import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import keras
from keras import layers
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =====================================================================
# 1. LOAD AND PREPARE 3-CHANNEL RGB DATA
# =====================================================================
print("--- Loading CIFAR-10 RGB Dataset ---")
(x_train, _), (x_test, _) = cifar10.load_data()

# Normalize pixel values to [0.0, 1.0]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Shape verification: CIFAR-10 images are 32x32 pixels with 3 channels (R, G, B)
# Expected Output: (50000, 32, 32, 3)
print(f"RGB Training Data Shape: {x_train.shape}")
print(f"RGB Testing Data Shape:  {x_test.shape}\n")


# =====================================================================
# 2. THE SAMPLING LAYER (Reparameterization Trick)
# =====================================================================
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# =====================================================================
# 3. BUILD THE RGB ENCODER & DECODER ARCHITECTURE
# =====================================================================
# For complex RGB objects, we expand the Latent Dimensions to 64 to capture colors, textures, and shapes.
latent_dim = 64

# --- RGB Encoder Network ---
encoder_inputs = keras.Input(shape=(32, 32, 3))  # Note the 3 channels here!
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)  # Extra layer for feature depth
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="rgb_encoder")

# --- RGB Decoder Network ---
decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(4 * 4 * 128, activation="relu")(decoder_inputs)  # Math match: 32 / 2 / 2 / 2 = 4
x = layers.Reshape((4, 4, 128))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)  # Upsamples to 8x8
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)  # Upsamples to 16x16
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)  # Upsamples to 32x32
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)  # Outputs 3 channels (RGB)

decoder = keras.Model(decoder_inputs, decoder_outputs, name="rgb_decoder")


# =====================================================================
# 4. CUSTOM COMPILATION AND LOSS PIPELINE FOR RGB
# =====================================================================
class RGB_VAE(keras.Model):
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

            # Binary Cross Entropy calculation aggregated over height, width, and ALL 3 color channels
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            # KL Divergence optimization constraint
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


# =====================================================================
# 5. EXECUTION & VISUALIZATION PIPELINE
# =====================================================================
if __name__ == "__main__":
    # Initialize and compile the model
    rgb_vae = RGB_VAE(encoder, decoder)
    rgb_vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))

    # Early stopping setup to preserve machine resources
    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=3,
        restore_best_weights=True
    )

    print("--- Starting RGB Autoencoder Training Loop ---")
    rgb_vae.fit(
        x_train,
        epochs=15,  # CIFAR-10 contains dense objects; 15-20 epochs show clean feature extractions
        batch_size=64,
        callbacks=[early_stop]
    )

    print("\n--- Visualizing Original vs VAE Reconstructed Real Objects ---")
    # Take a batch of real testing data to run through the reconstruction check
    _, _, test_encoded_z = rgb_vae.encoder.predict(x_test[:10])
    reconstructed_images = rgb_vae.decoder.predict(test_encoded_z)

    plt.figure(figsize=(20, 6))
    for i in range(10):
        # Display Original RGB Object
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(x_test[i])  # Matplotlib renders 3 channels directly out of the box!
        plt.title("Original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Regenerated Object Out of Latent Matrix Vectors
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(reconstructed_images[i])
        plt.title("Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.suptitle("RGB Image Compression & Extraction Blueprint", fontsize=16)
    plt.show()
