# Wie sieht die Apppoximation bei verschiedenen Modell aus.

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def f(x: float) -> float:

    #return x**2 + x + 10  # Einfache Funktion
    return x ** 4 + -5 * x ** 3 + 14 * x ** 2 + x + 10  # Schwerere Funktion


def relu(x: float) -> float:
    if x > 0:
        return x
    else:
        return 0


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(start=-10.0, stop=10.0, num=1000).reshape(-1, 1)
    y = f(x)
    return x, y


def build_model1() -> Sequential:  # Lineaeres Modeell ohne Aktivierungsfunktion
    model = Sequential()
    model.add(Dense(200))  # Input zu Hidden
    model.add(Dense(200))  # Input zu Hidden
    model.add(Dense(1))  # Vom Hidden zum Output
    return model


def build_model2() -> Sequential: # mit Relu Aktivierungsfunktion
    model = Sequential()
    model.add(Dense(500))  # Input zu Hidden
    model.add(Activation("relu"))  # ReLU vom Hidden
    model.add(Dense(500))  # Input zu Hidden
    model.add(Activation("relu"))  # ReLU vom Hidden
    model.add(Dense(500))  # Input zu Hidden
    model.add(Activation("relu"))  # ReLU vom Hidden
    model.add(Dense(1))  # Vom Hidden zum Output
    return model


def build_model3() -> Sequential: # mit Sigmoid Aktivierungsfunktion
    model = Sequential()
    model.add(Dense(500))  # Input zu Hidden
    model.add(Activation("sigmoid"))  # ReLU vom Hidden
    model.add(Dense(200))  # Input zu Hidden
    model.add(Activation("sigmoid"))  # ReLU vom Hidden
    model.add(Dense(500))  # Input zu Hidden
    model.add(Activation("sigmoid"))  # ReLU vom Hidden
    model.add(Dense(1))  # Vom Hidden zum Output
    return model


if __name__ == "__main__":
    x, y = get_dataset()

    ##################################
    # Betrachtung Modell 1
    ##################################
    model1 = build_model1()
    model1.compile(optimizer=Adam(learning_rate=1e-2),
                    loss="mse"
                   )
    model1.fit(x, y, epochs=30)
    y_pred_linear = model1.predict(x)

    ##################################
    # Betrachtung Modell 2
    ##################################
    model2 = build_model2()
    model2.compile(optimizer=Adam(learning_rate=1e-2),
                   loss="mse"
                   )
    model2.fit(x, y, epochs=30)
    y_pred_relu = model2.predict(x)

    ##################################
    # Betrachtung Modell 3
    ##################################
    model3 = build_model3()
    model3.compile(optimizer=Adam(learning_rate=10e-2),
                   loss="mse"
                  )
    model3.fit(x, y, epochs=30)
    y_pred_sigmoid = model3.predict(x)


    ##################################
    # Plot mit Subplots
    ##################################
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 8))
    plt.grid(True)
    ax1.plot(x, y, color="blue")
    ax1.plot(x.flatten(), y_pred_linear.flatten(), color="red")
    ax1.set_title('Ohne Aktivierungsfunktion (linear)')

    ax2.plot(x, y, color="blue")
    ax2.plot(x.flatten(), y_pred_sigmoid.flatten(), color="red")
    ax2.set_title('Sigmoid Aktivierungsfunktion')

    ax3.plot(x, y, color="blue")
    ax3.plot(x.flatten(), y_pred_relu.flatten(), color="red")
    ax3.set_title('Relu Aktivierungsfunktion')

    #legend = ax1.legend(loc='upper left')

    plt.show()
