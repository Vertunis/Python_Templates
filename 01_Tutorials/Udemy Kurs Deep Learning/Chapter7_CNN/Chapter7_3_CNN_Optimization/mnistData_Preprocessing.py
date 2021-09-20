# Export des Datasets mit Preprocessing als Klasse
# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/11776736#overview
# Data Augmentation (Zeile 51) https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/8641520#overview
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Neu für Bilder
from sklearn.model_selection import train_test_split

class MNIST:
    def __init__(self, with_normalization: bool = True):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Platzhalter für Aufteilung in train und validation Set (zusätzlch zum Testset, welches nur ganz am Ender verwendet werden soll)
        self.x_train_: np.ndarray = None
        self.y_train_: np.ndarray = None
        self.x_val_: np.ndarray = None
        self.y_val: np.ndarray = None
        self.val_size = 0
        self.train_splitted_size = 0
        # Preprocess x Data
        # Conv Layer erwarten Bildformat MIT Tiefendimension (auch wenn man graubilder hat) -> Expand Fkt.
        self.x_train = x_train.astype(np.float32) # Daten in float umwandeln (notwendig)
        self.x_train = np.expand_dims(x_train, axis=-1) # Erzeugt Tiefendimension. Hinter die Hinterste Achse (-1) würde Achse mit 1 hinzugefügt werden
        self.x_test = x_test.astype(np.float32)
        self.x_test = np.expand_dims(x_test, axis=-1)

        # Normalisierung der Daten
        if with_normalization:
            self.max_wert = 255
            self.x_train = self.x_train / self.max_wert
            self.x_test = self.x_test / self.max_wert

        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1] # 28
        self.width = self.x_train.shape[1] # 28
        self.height = self.x_train.shape[2] # 28
        self.depth = self.x_train.shape[3] # 1

        self.img_shape = (self.width, self.height, self.depth)

        self.num_classes = len(np.unique(y_train))  # 10
        # Preprocess y Data

        self.y_train = to_categorical(y_train, num_classes=self.num_classes, dtype=np.float32)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes, dtype=np.float32)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_splitted_train_validation_set(self, validation_size: float = 0.33) -> Tuple:
        self.x_train_, self.x_val_, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=validation_size)
        self.val_size = self.x_val_.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val_, self.y_train, self.y_val

    # NUR CNN´s: Erweiterung des Input Bildes mit ImageDataGenerator
    def data_augmentation(self, augment_size: int = 5_000) -> None:
        image_generator = ImageDataGenerator(
            rotation_range=5,  # In Grad
            zoom_range=0.08,  # Heranzoomen im Bild, Prozentual auf die Pixel gerechnet (28 Pixel*0,08 => 2 Pixel)
            width_shift_range=0.08,  # Prozentual auf die Pixel gerechnet
            height_shift_range=0.08  # Prozentual auf die Pixel gerechnet
        )
        # Fit the Data Generator
        image_generator.fit(self.x_train, augment=True)  # Wird NUR auf des Testset angewendet
        # Get Random Train images for the data augmentation
        rand_idx = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idx].copy() # Copy, da sonst ursprüngliches Array kopiert werden würde
        y_augmented = self.y_train[rand_idx].copy() # Y-Daten verändern sich nicht, jedoch werden die entsprechenden Random indize-Werte mit übernommen

        # Beispielbild vorher ausgeben
        plt.imshow(x_augmented[0, :, :, 0], cmap="gray")
        plt.show()

        # In folgender zeile findet die eigntliche Augmentation statt
        x_augmented = image_generator.flow(
            x_augmented,
            np.zeros(augment_size),
            batch_size=augment_size,  # Für alle 5000 Bilder soll das passieren
            shuffle=False # Shufflen wollen wir nicht
        ).next()[0]  # -> Komplettes Array mit den augmented Bilder aufrufen

        # Beispielbild nachher ausgeben
        plt.imshow(x_augmented[0, :, :, 0], cmap="gray")
        plt.show()

        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented)) # Vektor der leicht veränderten Bilder angehängt
        self.y_train = np.concatenate((self.y_train, y_augmented)) # Vektor der leicht veränderten Bilder angehängt
        self.train_size = self.x_train.shape[0]


if __name__ == "__main__":
    data = MNIST()
    print(data.train_size)
    print(data.test_size)
    print(data.x_test.shape)
    print(data.y_test.shape)
    print(f"Min of x_train: {np.min(data.x_train)}")
    print(f"Max of x_train: {np.max(data.x_train)}")
    data.data_augmentation(augment_size=5_000)

