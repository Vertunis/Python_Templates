# Model API: Build Model nicht mehr mit Sequential sondern mit Model API. Notwendig wenn man mal mehr als einen Input haben sollte
# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/8706484#overview

import os

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import TensorBoard
from typing import Tuple

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model # NEU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# https://www.tensorflow.org/api_docs/python/tf/keras
# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/17300996#overview


MODELS_DIR = os.path.abspath(r"M:\Google Drive\Programming\Python\01_Tutorials\Udemy Kurs Deep Learning\models")  # Speicherort für Modelle
MODELS_FILE_PATH = os.path.join(MODELS_DIR, "mnist_cnn5.h5")  # h5 Datei wird von Keras zum speichern der Gewichte verwendet

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

LOGS_DIR = os.path.abspath(r"M:\Google Drive\Programming\Python\01_Tutorials\Udemy Kurs Deep Learning\logs")
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_cnn5")  # Muss einzigartiger name sein

# Entspechend mnistCnnModel_Create_2
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


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Model:

    input_img = Input(shape=img_shape) # Neuer Input des Bildes

    # Erläuterung für x
    #x = Conv2D(filters=32, kernel_size=3, padding="same", strides=1) # Objekt erstellen
    #x(input_img) # Übergabe des inputs an des bereits erstellte Objekt
    x = Conv2D(filters=32, kernel_size=3, padding="same", strides=1)(input_img) # Objekt Activation Layer erstellen und Input Layer übergeben
    x = Activation("relu")(x) # Objekt Activation Layer erstellen und letzten Layer übergeben
    x = Conv2D(filters=32, kernel_size=3, padding="same", strides=1)(x) # Entspricht dem obigen ausgeklammerten in Kurzform
    x = Activation("relu")(x) # Objekt Activation Layer erstellen und letzten Layer übergeben
    x = MaxPooling2D()(x)

    x = Conv2D(filters=32, kernel_size=3, padding="same", strides=1)(x) # Objekt Activation Layer erstellen und Input Layer übergeben
    x = Activation("relu")(x) # Objekt Activation Layer erstellen und letzten Layer übergeben
    x = Conv2D(filters=32, kernel_size=3, padding="same", strides=1)(x) # Entspricht dem obigen ausgeklammerten in Kurzform
    x = Activation("relu")(x) # Objekt Activation Layer erstellen und letzten Layer übergeben
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(inputs=[input_img],
                  outputs=[y_pred])

    model.summary()
    return model

def plot_filters():

    print("Ausgabe der Filter Details")
    first_conv_layer = model.layers[1]  # Greift auf unserem erstem Conv_Layer (Conv2D) aus Build_Model zu. An stelle [0] wäre Input
    layer_weights = first_conv_layer.get_weights()
    print(layer_weights)
    kernels = layer_weights[0]
    print(kernels.shape)  # Ergibt sich ausKerneldimension(L*B*Tiefe)*FilterAnzahl -> zb. (3*3*1)*32

    num_filters = kernels.shape[3]  # Anzahl Filter
    subplot_grid = (num_filters // 4, 4)
    fig, ax = plt.subplots(subplot_grid[0], subplot_grid[1], figsize=(20, 20))  # Definiton der Teilgrafiken
    ax = ax.reshape(num_filters)

    for filter_idx in range(num_filters):
        ax[filter_idx].imshow(kernels[:, :, 0, filter_idx], cmap="gray")

    ax = ax.reshape(subplot_grid)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":

    img_shape = (28, 28, 1)  # Anzahl der Features kommt aus Dataset: 60000 Bilder a 28x28 Pixel = 784Pixel (Muss für Dense Funktion als Vektor vorliegen)
    num_targets = 10  # Anzahl der Klassen (Wird auch für ONEHOT Vektor verwendet)

    (x_train, y_train), (x_test, y_test) = prepare_dataset(num_targets)  # Funktionsaufruf für Dataset

    # Schritt 1:  Modell erstellen
    model = build_model(img_shape, num_targets)

    # Schritt 2: Das Modell muss kompiliert werden. Hier werden die Gewichte initialisiert. Besteht aus Fehlerfunktion, Optimizer und einer Metric
    model.compile(
        loss="categorical_crossentropy", # categorical_crossentropy für Klassifikationsprobleme > 2 Klassen. Übergabe als String (nur wenn in TF existent. Dabei werden immer Defaultwert genommen) oder alternativ als Funktionsaufruf: tf.keras.losses.categorical_crossentropy
        #optimizer="Adam", # Übergabe als String (nur wenn in TF existent. Dabei werden immer Defaultwert genommen) oder alternativ als Funktionsaufruf:  tf.keras.optimizers.Adam
        optimizer = Adam(learning_rate=0.0005), # Objekt des Adam Optimizers mit Learnrate (Default 0.001)
        metrics=["accuracy"] # (OPTIONAL) Wichtig: nicht die Funktion aufrufen, sondern das Funktionsobjekt übergben. Wichtig TF erwartet hier immer eine Liste
    )

    # Zwischenschritt: Tensorboard Objekt erzeugen -> Wichtig muss in der model.fit Methode mit übergeben werden (ganz unten)
    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,  # Wichtig Pfad für Log-directoy angeben
        histogram_freq=1,
        write_graph=True)

    # Schritt 3: Training ausführen
    model.fit(
        x=x_train,
        y=y_train,
        epochs=10,                        # (OPTIONAL) Wieviele Epochen sollen trainiert werden. Default = 11
        batch_size=128,                   # (OPTIONAL) Number of samples per batch of computation. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of a dataset, generators, or keras.utils.Sequence instances (since they generate batches)
        verbose=1,                        # (OPTIONAL) Wenn Verbose =1, dann wird der Output jede Epoche ausgegeben
        validation_data=(x_test, y_test), # (OPTIONAL) Validation Daten
        callbacks=[tb_callback]           # Nur wenn TensorCallbacks implementiert werden verwenden -> Liste von Callbacks die wir hinzufügen wollen
    )

    # Schritt 4 Testing (Optional). Die Evaluate Funktion returned einen Losswert und einen Metric-Wert
    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0      # (OPTIONAL) Wenn Verbose =1, dann wird der Output ausgegeben
    )
    print(f"scores before saving: {scores}")

    # ...........
    # Speichern und Laden der Weights des Modells. Wichtig es muss in das Gleiche Modell geladen werden, damit die Weights wieder funktionieren. Spricch gleiche Anzahl an Neurone in und Outputs
    # Nach dem Speichern und Laden des Modells kann das Training und das Speichern auskommentiert werden
    # ...........
    model.save_weights(filepath=MODELS_FILE_PATH) # Speichern der Weights als h5 Datei unter Fold er Models
    model.load_weights(filepath=MODELS_FILE_PATH)
    # Schritt 4.2 Testing (Optional). Die Evaluate Funktion returned einen Losswert und einen Metric-Wert
    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0      # (OPTIONAL) Wenn Verbose =1, dann wird der Output ausgegeben
    )
    print(f"scores after loading: {scores}")

    plot_filters()