# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/12684241#overview

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


def conv2D(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    rows, cols = image.shape  # 28x28
    k_size = kernel.shape  # 2x2 Kernelgröße
    conv_image =np.zeros(shape=(rows, cols), dtype=np.float32)  # Ergebnisbild 28x28

    for i in range(rows - k_size[0]): # Über jeden Pixel des Bildes iterieren, jedoch müssen wir im Range bleiben daher minus k_size
        for j in range(cols - k_size[1]):
            conv_image[i][j] = np.sum(kernel*image[i:i+k_size[0], j:j+k_size[1]]) # Faltungsvorgang. Summe der Kompletten Matrix (nicht nur entlang einer Achsse)
                                                                              # Intuition: Je größer der Wert zwischen Kernel und Aktuellem Bildauschnitt desto ähnlicher der Bildausschnitt zu dem Kernel
    return conv_image


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image = x_train[0]
    image = image.reshape((28, 28))
    kernel = np.random.uniform(low=0.0, high=1.0, size=(2, 2))

    conv_image = conv2D(image, kernel)

    plt.imshow(image, cmap="gray")
    plt.show()

    plt.imshow(conv_image, cmap="gray")
    plt.show()
