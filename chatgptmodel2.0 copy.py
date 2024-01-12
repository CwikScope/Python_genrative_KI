import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Passe den Pfad entsprechend an
datenpfad = "C:/Users/timon/downloads/testbilder"
daten_generator = ImageDataGenerator(rescale=1./255)

def custom_image_data_generator(directory, batch_size=1, target_size=(256, 256)):
    image_generator = daten_generator.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,  # Setze class_mode auf None, um keine Klassenlabels zu erhalten
        shuffle=True
    )

    while True:
        # Erhalte Bilder und erstelle Dummy-Zielvariablen
        images = image_generator.next()
        dummy_labels = (1, 256, 256, 3)

        yield images, dummy_labels

# Verwende den benutzerdefinierten Generator
daten_strom = custom_image_data_generator(datenpfad)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

generator = Sequential([
    Dense(256 * 256 * 3, input_shape=(100,), activation='relu'),
    #Reshape((256, 256, 3)),
    Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same'),
    Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
    Conv2D(3, kernel_size=3, activation='tanh', padding='same')  # 3 steht für RGB-Kanäle
])

generator.compile(loss='binary_crossentropy', optimizer='adam')

generator.fit(daten_strom, epochs=50, steps_per_epoch=1)

# Generiere zufälligen Input
random_input = np.random.rand(1, 100)
# Generiere ein Bild
generated_image = generator.predict(random_input)

# Zeige das generierte Bild an
plt.imshow(generated_image[0])
plt.show()

