# Vergleich der Modelle In dieser Datei wird die Normalsierung in mnistData_Preprocessing.py aktiviert
# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/23299958#overview

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

from mnistData_Preprocessing import MNIST

# https://www.tensorflow.org/api_docs/python/tf/keras
# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/17300996#overview


MODELS_DIR = os.path.abspath(r"M:\Google Drive\Programming\Python\01_Tutorials\Udemy Kurs Deep Learning\models\Chapter7_3")  # Speicherort für Modelle
MODELS_FILE_PATH = os.path.join(MODELS_DIR, "mnist_7_3_cnn2.h5")  # h5 Datei wird von Keras zum speichern der Gewichte verwendet

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

LOGS_DIR = os.path.abspath(r"M:\Google Drive\Programming\Python\01_Tutorials\Udemy Kurs Deep Learning\logs\Chapter7_3")
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "cnn_norm_augment")  # Muss einzigartiger name sein


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


if __name__ == "__main__":

    data = MNIST(with_normalization=False)
    data.data_augmentation(augment_size=5_000)  # Wichtig die Augmentation der Daten vor dem Split

    x_train_, x_val_, y_train_, y_val_ = data.get_splitted_train_validation_set()

    # Schritt 1:  Modell erstellen
    model = build_model(data.img_shape, data.num_classes)

    # Schritt 2: Das Modell muss kompiliert werden. Hier werden die Gewichte initialisiert. Besteht aus Fehlerfunktion, Optimizer und einer Metric
    model.compile(
        loss="categorical_crossentropy", # categorical_crossentropy für Klassifikationsprobleme > 2 Klassen. Übergabe als String (nur wenn in TF existent. Dabei werden immer Defaultwert genommen) oder alternativ als Funktionsaufruf: tf.keras.losses.categorical_crossentropy
        #optimizer="Adam", # Übergabe als String (nur wenn in TF existent. Dabei werden immer Defaultwert genommen) oder alternativ als Funktionsaufruf:  tf.keras.optimizers.Adam
        optimizer=Adam(learning_rate=0.0005), # Objekt des Adam Optimizers mit Learnrate (Default 0.001)
        metrics=["accuracy"] # (OPTIONAL) Wichtig: nicht die Funktion aufrufen, sondern das Funktionsobjekt übergben. Wichtig TF erwartet hier immer eine Liste
    )

    # Zwischenschritt: Tensorboard Objekt erzeugen -> Wichtig muss in der model.fit Methode mit übergeben werden (ganz unten)
    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        write_graph=True
    )

    # Schritt 3: Training ausführen
    model.fit(
        x=x_train_,
        y=y_train_,
        epochs=40,                        # (OPTIONAL) Wieviele Epochen sollen trainiert werden. Default = 11
        batch_size=128,                   # (OPTIONAL) Number of samples per batch of computation. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of a dataset, generators, or keras.utils.Sequence instances (since they generate batches)
        verbose=0,                        # (OPTIONAL) Wenn Verbose =1, dann wird der Output jede Epoche ausgegeben
        validation_data=(x_val_, y_val_), # (OPTIONAL) Validation Daten
        callbacks=[tb_callback]           # Nur wenn TensorCallbacks implementiert werden verwenden -> Liste von Callbacks die wir hinzufügen wollen
    )

    '''
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
    '''
