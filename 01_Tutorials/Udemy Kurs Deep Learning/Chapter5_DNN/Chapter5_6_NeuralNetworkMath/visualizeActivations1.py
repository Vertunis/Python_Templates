import matplotlib.pyplot as plt
import numpy as np


def relu(x: float) -> float:
    if x > 0:
        return x
    else:
        return 0


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # y = ReLU(wx + b)
    # y = sigmoid(wx + b)
    # shift = -b/w
    w = 1
    b = -4
    # shift = 2 ; Shift der Aktivierungsfunktion im Bild
    act_fn = sigmoid

    x = np.linspace(start=-10, stop=10, num=5000)
    y_act = np.array([act_fn(xi * w + b) for xi in x])
    y = np.array([act_fn(xi * 1 + 0) for xi in x])

    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.plot(x, y, color="blue", label='Standard (b=1, W=1)') # Standard Relu
    plt.plot(x, y_act, color="red",  label=f'Modifiziert (b={b}, W={w})') # Modifizierte

    plt.plot((0,-b/w),(0.5,0.5), c='black',linestyle="--", label='Shift: -b/w') # Plot des Shifts

    plt.legend(loc='lower right')

    plt.show()
