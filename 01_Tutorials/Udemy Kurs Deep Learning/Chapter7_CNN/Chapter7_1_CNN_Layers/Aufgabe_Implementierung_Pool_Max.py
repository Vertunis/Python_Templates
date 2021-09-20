# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/quiz/4457514#overview
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

def get_kernel_values(i: int, j: int, image = np.ndarray) -> np.ndarray:  # Hilfsfunktion
    kernel_values = image[i: i+2, j: j+2] # Nimmt die aktuellen Werte von i bis i+2

    return kernel_values

def max_pooling(image: np.ndarray) -> np.ndarray:
    # Setup output image as ndnarray -> Wie groß soll das Output Bildsein
    #(variable) new_cols: Any // 2 #
    new_rows = image.shape[0] // 2 # Integer Division
    new_cols = image.shape[1] // 2 # Integer Division
    output = np.zeros(shape=(new_rows, new_cols), dtype=np.float32) # Ergebnisdimension entsprechend schon mal mit nullen
    #Compute the Values
    for i in range(new_rows):
        for j in range(new_cols):
            kernel_values = get_kernel_values(2*i, 2*j, image)
            max_val = np.max(kernel_values.flatten())
            output[i][j] = max_val
    return output
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image = x_train[0]
    image = image.reshape((28, 28))

    pooling_image = max_pooling(image)

    print(image.shape)
    print(pooling_image.shape)

    plt.imshow(image, cmap="gray")
    plt.show()

    plt.imshow(pooling_image, cmap="gray")
    plt.show()
