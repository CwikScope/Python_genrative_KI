import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam

print("Importieren funktioniert")

from tensorflow.keras.datasets import mnist

(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5 ) / 127.5
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

def build_generator(latent_dim):
    model = Sequential ()
    model.add(Dense(128*7*7, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(128, (4, 4 ), strider=(2, 2), padding='same'))
    model.add(LeakyReLu(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(1, (4, 4), strider=(2, 2), padding='same', activation='tanh'))
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLu(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

latent_dim = 100
img_shape = (28, 28, 1)

generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

def train_gan(generatorm, discriminator, gan, X_train, latent_dim, epochs=10000, batch_size=128):
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        fake_imgs = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch((fake_imgs, fake_labels))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, valid_labels)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

train_gan(generator, discriminator, gan, X_train, latent_dim, epochs=3000, batch_size=64)

def generate_images(generator, latent_dim, n_samples=16):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated_imgs = generator.predict(noise)

    plt.figure(figsize=(4, 4))
    for i in range(generated_imgs.shape[0]):
        plt.subplot(4, 4, i +1)
        plt.imshow(generated_imgs[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

generate_images(generator, latent_dim)