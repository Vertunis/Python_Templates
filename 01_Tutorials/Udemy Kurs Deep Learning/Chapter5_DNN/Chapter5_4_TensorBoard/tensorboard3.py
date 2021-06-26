# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/18345258#overview
# Einbau Confusion Matrix aus Utils/tf_utils auf Basis von tf.keras.callbacks.Callback
# Nach jeder Epoche soll eine plot_confusion_matrix erstellt werden

import os

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from tf_utils.callbacks import ConfusionMatrix # Neu -> Liegt unter Utils/tf_utils


MODELS_DIR = os.path.abspath(r"M:\Google Drive\Programming\Python\01_Tutorials\Udemy Kurs Deep Learning\models")  # Speicherort für Modelle
MODELS_FILE_PATH = os.path.join(MODELS_DIR, "mnist_model.h5")  # h5 Datei wird von Keras zum speichern der Gewichte verwendet

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

LOGS_DIR = os.path.abspath(r"M:\Google Drive\Programming\Python\01_Tutorials\Udemy Kurs Deep Learning\logs")
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_model_cm")

if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def prepare_dataset(num_features: int, num_classes: int) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)

    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_classes: int) -> Sequential:
    init_w = TruncatedNormal(mean=0.0, stddev=0.01)
    init_b = Constant(value=0.0)

    model = Sequential()
    model.add(Dense(units=500, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,),))
    model.add(Activation("relu"))
    model.add(Dense(units=300, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=50, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=num_classes, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("softmax"))
    model.summary()

    return model


if __name__ == "__main__":
    num_features = 784
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(num_features, num_classes)

    optimizer = Adam(learning_rate=0.001)
    epochs = 2
    batch_size = 256

    model = build_model(num_features, num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        histogram_freq=1,
        write_graph=True
    )

    classes_list = [class_idx for class_idx in range(num_classes)]

    cm_callback = ConfusionMatrix(               # Neu aufruf der Klasse Confusion Matrix
        model,
        x_test,
        y_test,
        classes_list=classes_list,
        log_dir=MODEL_LOG_DIR
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback, cm_callback], # Hier wird der Callback ins Modell gepackt
    )

    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0
    )
    print("Scores: ", scores)
