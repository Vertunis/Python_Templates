# Doku https://www.tensorflow.org/api_docs/python/tf/keras

from typing import Tuple

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def prepare_dataset(num_classes: int) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Conv Layer erwarten Bildformat MIT Tiefendimension (auch wenn man graubilder hat) -> Expand Fkt.
    x_train = x_train.astype(np.float32) # Daten in float umwandeln (notwendig)
    x_train = np.expand_dims(x_train, axis=-1) # Erzeugt Tiefendimension. Hinter die Hinterste Achse (-1) würde Achse mit 1 hinzugefügt werden
    x_test = x_test.astype(np.float32)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    return (x_train, y_train), (x_test, y_test)


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    model = Sequential()

    # Wichtig: Nach jedem Conv oder Dense Layer einen Activation Layer einbauen !
    model.add(Conv2D(filters=16, kernel_size=3, padding="same", strides=1, input_shape=img_shape))  # Filter + kernel  muss bei Conv2D immer angegeben werden
    model.add(Activation("relu"))

    model.add(MaxPooling2D()) # Reduzierung der Komplexität des bildes -> 3 Parameter: Poolsize(Größe des Poolings); Strides = 2 (Man will doppelte Werte vermeiden, daher 2)-> Defaultwerte siehe Doku

    model.add(Conv2D(filters=16, kernel_size=3))  # Filter + kernel  muss bei Conv2D immer angegeben werden
    model.add(Activation("relu"))

    model.add(MaxPooling2D()) # Reduzierung der Komplexität des bildes -> 3 Parameter: Poolsize(Größe des Poolings); Strides = 2

    # model.add(Reshape(target_shape=26*26*16)) # Alternativ zu unterer Flatten Funktion. Damit wir vom Bildformat in Vektorformat kommen (Jeodch lieber flatten nehmen!!!!)
    model.add(Flatten())
    model.add(Dense(units=num_classes))
    model.add(Activation("softmax"))
    model.summary()

    return model


if __name__ == "__main__":
    img_shape = (28, 28, 1)
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(num_classes)

    model = build_model(img_shape, num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"]
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=10,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0
    )
    print(scores)
