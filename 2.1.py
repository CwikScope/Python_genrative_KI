import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Passe den Pfad entsprechend an
datenpfad = "C:/Users/timon/downloads/testbilder"
daten_generator = ImageDataGenerator(rescale=1./255)

daten_strom = daten_generator.flow_from_directory(
    datenpfad,
    target_size=(256, 256),
    batch_size=1,
    class_mode=None,
    shuffle=True
)

for batch in daten_strom:
    print(batch.shape)
    break  # nur den ersten Batch ausgeben

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

generator = Sequential([
    Dense(256 * 256 * 3, input_shape=(100,), activation='relu'),
    Reshape((256, 256, 3)),
    Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same'),
    Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
    Conv2D(3, kernel_size=3, activation='tanh', padding='same')
])

generator.compile(loss='binary_crossentropy', optimizer='adam')

# Verwende fit anstelle von fit_generator
# Hier gehen wir davon aus, dass dein Generator auch die Zielvariablen liefert (was normalerweise der Fall ist)
generator.fit(daten_strom, epochs=50, steps_per_epoch=len(daten_strom), target_data=(daten_strom))
