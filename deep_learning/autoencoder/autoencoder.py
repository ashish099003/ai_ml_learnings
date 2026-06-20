import ssl

from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_unverified_context
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
# Visualize the outputs
import matplotlib.pyplot as plt


def autoencoder_mirror():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape)
    print(x_train[0])

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    print(x_train[0])

    #Reshaping the images to 1D vectors


    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    input_img = keras.Input(shape=(784,))
    encoded = layers.Dense(128, activation='relu')(input_img)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)


    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(784, activation='sigmoid')(decoded)


    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))




    decoded_imgs = autoencoder.predict(x_test)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def denoising():

    (x_train, _), (x_test,_) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255


    # Reshaping the images to 1D vectors

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)
    x_test_noisy = x_test + noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    print(x_train.shape)
    print(x_train_noisy.shape)
    print(x_test.shape)
    print(x_test_noisy.shape)

    n = 10
    plt.figure(figsize=(20, 2))
    for i in range(1, n + 1):
        ax = plt.subplot(1, n, i)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    # AutoEncoder model
    input_img = keras.Input(shape=(784,))
    encoded = layers.Dense(128, activation='relu')(input_img)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)

    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(784, activation='sigmoid')(decoded)

    # Compile and Fit
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(x_train_noisy, x_train,  # NOTE: input is noisy, output is non-noisy
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))

    decoded_imgs = autoencoder.predict(x_test_noisy)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# def autoencoder_ratings():
#     print("movie rating with autoencoder")
#     ratings = pd.read_csv('/Users/ashish/PycharmProjects/pythonTest/ai_ml_learnings/deep_learning/autoencoder/ratings.csv')
#     print(ratings)
#     rm = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
#     print(rm.head())
#     train, val = train_test_split(rm, test_size=0.2)
#     X_train = train.values
#     X_val = val.values
#     input_movie = keras.Input(shape=(668,))
#     encoded = layers.Dense(512, activation='relu')(input_movie)
#     encoded = layers.Dense(256, activation='relu')(encoded)
#     encoded = layers.Dense(128, activation='relu')(encoded)
#
#     decoded = layers.Dense(256, activation='relu')(encoded)
#     decoded = layers.Dense(512, activation='relu')(decoded)
#     decoded = layers.Dense(668, activation='linear')(decoded)
#     autoencoder = keras.Model(input_movie, decoded)
#     autoencoder.compile(optimizer='adam', loss='mean_squared_error')
#     autoencoder.fit(X_train, X_train,
#                     epochs=100,
#                     batch_size=256,
#                     shuffle=True,
#                     validation_data=(X_val, X_val))




if __name__=='__main__':
    denoising()
    # autoencoder_ratings()
