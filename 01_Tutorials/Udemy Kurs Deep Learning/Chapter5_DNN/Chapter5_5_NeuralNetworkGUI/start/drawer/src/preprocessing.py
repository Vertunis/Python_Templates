# In dieser Datei wird das gemalte Bild geladen und Preprocessed (Centered, NNormalised, Resized)
# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/9094758#overview

import os
from typing import Any

import cv2  # Führende Bibliothek für Bildverarbeitung
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import center_of_mass


FILE_PATH = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(os.path.dirname(FILE_PATH))


def load(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Bild laden: jpeg -> numpyarray -> in Grau.
    return image


def resize(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (28, 28))  # Reshape in Tupel 28x28. Wichtig: 2 Dimensional für CNN auch anwendbar
    return image


def normalize(image: np.ndarray) -> np.ndarray:
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV) # Binäres Bild nur schwarz Weiss und wir togglen die Werte noch. Da unser Bild ist schwarz auf Weiss. Das MNIST mit welchem Trainiert wurde ist aber weiß auf schwarz
    return image


# Bild Zentrieren
def center(image: np.ndarray) -> np.ndarray:
    cy, cx = center_of_mass(image)
    rows, cols = image.shape
    shift_x = np.round(cols/2 - cx).astype(int)
    shift_y = np.round(rows/2 - cy).astype(int)
    M = np.array([[1, 0, shift_x], [0, 1, shift_y]]).astype(np.float32)  # Affine Transformation
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

# Hauptmethode mit Übernahme des Selbst gemalten Bildes
def get_image(DrawingFrame: Any, debug: bool = False) -> np.ndarray:
    pixmap = DrawingFrame.grab()
    temp_image_file_path = os.path.join(PROJECT_DIR, "ressources", "imgs", "temp_image.jpg") # Speicher-Pfad deklarieren
    pixmap.save(temp_image_file_path)  # Im "imgs" Ordner wird das selbst gemalte Bild abgespeichert
    image = load(temp_image_file_path)
    image = resize(image)
    image = normalize(image) # ACHTUNG: Unser Bild ist schwarz auf Weiss. Das MNIST mit welchem Trainiert wurde ist aber weiß auf schwarz. Daher werden Werte getogeelt
    image = center(image)

    # Eigenes Bild nochmal anzeigen
    plt.imshow(image, cmap="gray")
    plt.show()

    return image
