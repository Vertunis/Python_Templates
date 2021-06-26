# In dieser Datei wird ein Keras Modell erstellt, welches Grafisch die Aktivierungsfunktionen darstellen. Dabei werden mehrere Gewichte betrachtet

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def f(x: float) -> float:
    return x ** 2 + x + 10


def relu(x: float) -> float:
    if x > 0:
        return x
    else:
        return 0


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(start=-10.0, stop=10.0, num=1000).reshape(-1, 1) # Reshape für Keras für Batch-format -> Werte von -10 bis 10
    # print(x.shape)
    y = f(x)
    return x, y


def build_model() -> Sequential:
    model = Sequential()
    model.add(Dense(12))  # Input zu Hidden (Dense Layer mit 12 Neuronen)
    model.add(Activation("relu"))  # ReLU vom Hidden
    model.add(Dense(1))  # Vom Hidden zum Output. Bei der Regression in der Regel keine Aktivierungsfunktion am Ausgang
    return model


if __name__ == "__main__":
    x, y = get_dataset()

    model = build_model()

    model.compile(
        optimizer=Adam(learning_rate=5e-2),
        loss="mse"
    )
    model.fit(x, y, epochs=20)

    W, b = model.layers[0].get_weights() # Ausgabe der Weights und des Bias am Stelle [0] -> erster Layer mit 12 Neuronen (siehe oben)
    W2, b2 = model.layers[2].get_weights() # Ausgabe der Weights und des Bias am Stelle [2] -> Dense Layer mit 1 Neuronen (siehe oben)
    print(W.shape, b.shape) # Size W(1,12) , b(12,1)
    print(W2.shape, b2.shape) # Size(12,1)

    # Umformen in Vektor mittels flatten Funktion https://www.w3resource.com/numpy/manipulation/ndarray-flatten.php
    W = W.flatten()
    W2 = W2.flatten()
    b = b.flatten()
    b2 = b2.flatten()
    print(W.shape, b.shape) # Size W(1,12) , b(12,1)
    print(W2.shape, b2.shape) # Size(12,1)

    # Nachbau des Modells um an Werte im Modell zu kommen -> Berechnung der Gewichtsmatrix mit HiddenLayer + Bias
    # [1, 2, ...., 12] Wir iterieren über alle Gewichte
    # yhi -> y Wert aus Hidden Layer, yri -> y_Werte aus relu Funktion
    for i in range(1, len(W) + 1):
        y_hidden = np.array([W[:i] * xi + b[:i] for xi in x]) # Wir geben 1000 Werte (siehe Dataset) ins Netzwerk und schauen was die Hidden Neuronen ausgeben. Wir wollen jedoch nur die ersten i Gewichte haben
        y_relu = np.array([[relu(yhi) for yhi in yh] for yh in y_hidden])  # Wert der Relu Aktivierungsfunktion
        y_output = np.array([np.dot(W2[:i], yri) + b2 for yri in y_relu])  # Wert des Output Neurons

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
        plt.title("Num weights: " + str(i))
        plt.grid(True)
        ax1.plot(x, y, color="blue", label='zu approximierendes Dataset')  # Plottet das Dataset
        ax1.plot(x.flatten(), y_output.flatten(), color="red", label= f'Approximiertes Dataset (Output NN) mit {i} Weights') # Mit Flatten die Matrizen in Vektoren umwandeln für plt Funktion
        ax2.plot(x, y_relu.T[-1], label= f'Output Relu-Layer') # Transponieren und von der Matrix immer den letzten Eintrag nehmen

        legend = ax1.legend(loc='upper left')
        legend = ax2.legend(loc='upper left')

        #plt.legend(loc='upper left')
        plt.show()
        plt.close()
