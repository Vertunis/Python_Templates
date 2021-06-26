# Entspricht fast dem Code aus:
# ./Udemy Kurs Deep Learning/Chapter5_DNN/Chapter5_3_MNISTClassification/mnistKerasDnn_mit_Speicherung_Gewichte.py
# nn_predict(), main_GUI() ist neu

# Wichtig allererstes mal muss diesee DNN Datei selber aufgerufen werden
import os

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import TruncatedNormal  # Initialisierung für Gewichte
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam  # Optimizer
from tensorflow.keras.utils import to_categorical  # Für OneHot Vektor

# https://www.tensorflow.org/api_docs/python/tf/keras
# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/12664676#overview
# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/23254514#overview
# # https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/8515640#overview


FILE_PATH = os.path.abspath(__file__)  # Ermittelt kompletten Pfad zu aktuellem File mit Filename und Endung
PROJECT_DIR = os.path.dirname(os.path.dirname(FILE_PATH)) # Extrahiert nur den Pfad
MODEL_FILE_PATH = os.path.join(PROJECT_DIR, "ressources", "weights", "dnn_mnist.h5") # Speicherort für Modelle

# Umformatierung unserer In-/Output Daten aus MNIST Dataset
def prepare_dataset(num_features: int, num_targets: int) -> tuple:

    # Test Train Split
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Lädt Datensatz aus Inet herunter

    # Umformatierung der Datentypen nach Float32
    x_train = x_train.reshape(-1, num_features).astype(np.float32) # Zwingt das Format in eine Spalte  -> Keras erwartet Shape.(X, 1)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)

    y_train = to_categorical(y_train, num_classes=num_targets, dtype=np.float32) # ONEHOT Vektor. Klassifikation kommen duetlich besser klar wenn die y-Werte als OneHot Vektor vorliegen
    y_test = to_categorical(y_test, num_classes=num_targets, dtype=np.float32)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)  # Doppeltes Tupel

# Model mit bestimmter Anzahl an Layern erzeugen -> Returned wird das Sequential Objekt
def build_model(num_features: int, num_targets: int) -> Sequential:  # Inpput: num_features ; Output num_targets
    init_w = TruncatedNormal(mean=0.0, stddev=0.01)  # Startwert für Gewicht deklarieeren. Zufallsverteilung um 0 mit Standardabweichung 0.01
    init_b = Constant(value=0.0)  # Initialisierung bias Neuronen. INFO: Per Default werden bias Neuronen immer mit 0 in keras initialisiert

    # Modellstruktur erstellen. Der Output des ersten Layers ist der Input des folgenden Layers
    model = Sequential()  # Objekt der Sequential Klasse erstellen
    # Beim ersten Layer MUSS Input Shape deklariert werden
    model.add(Dense(units=500, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,))) # Layer hinzufügen -> Erinnerung  Dense Layer: Hidden Layer mit Input Layer Verbinden. Units -> Wieviele Neuronen hat der Hidden Layer, input_shape -> Shape der Eingangsfeatures
    model.add(Activation("relu")) # Aktivierungsfunktion verknüpfen
    model.add(Dense(units=250, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=num_targets, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("softmax")) # WICHTIG bei Klassifikation immer SOFTMAX-Aktivierungsfunktion -> Outputwerte werden als Wahrscheinlichkeiten ausgegeben
                                     # WICHTIG bei Regression keine SOFTMAX am Ende
    model.summary()  # Ausgabe der Struktur des Netzwerkes. Erinnerung Activation Funktionen haben 0 Parameter. Die Anzahl der Parameter ergibt sich aus (Anz Input* Anz Neuronen)+ Anz Bias Einträge

    return model


# NEU Predict. Hier wird da zu Analysierdende Bild verarbeitet
def nn_predict(model: Sequential, image: np.ndarray = None) -> int:
    if image is not None:
        y_pred = model.predict(image.reshape(1, 784))[0] # Erwarteter Shape für Bild für unser Dense Netzwerk. Dieses 1 Steht für die Batch-Dimension, dass wir quasi nur ein Bild mit reinnehmen
                                                         # y_pred ist eine Liste in der ein Onehot-Array liegt -> mit [0] Zugriff auf Onehot Array
        y_pred_class_idx = np.argmax(y_pred, axis=0)     # Index des Maximalwertes in OneHot ermitteln
        return y_pred_class_idx
    else:                           # Wenn kein Gültiges Bild -1 -> Ready to Detect Bild
        return -1

def nn_train() -> None:
    num_features = 784  # Anzahl der Features kommt aus Dataset: 60000 Bilder a 28x28 Pixel = 784Pixel (Muss für Dense Funktion als Vektor vorliegen)
    num_targets = 10    # Anzahl der Klassen (Wird auch für ONEHOT Vektor verwendet)

    (x_train, y_train), (x_test, y_test) = prepare_dataset(num_features, num_targets)  # Funktionsaufruf für Dataset

    # Schritt 1:  Modell erstellen
    model = build_model(num_features, num_targets)

    # Schritt 2: Das Modell muss kompiliert werden. Hier werden die Gewichte initialisiert. Besteht aus Fehlerfunktion, Optimizer und einer Metric
    model.compile(
        loss="categorical_crossentropy",  # categorical_crossentropy für Klassifikationsprobleme > 2 Klassen. Übergabe als String (nur wenn in TF existent. Dabei werden immer Defaultwert genommen) oder alternativ als Funktionsaufruf: tf.keras.losses.categorical_crossentropy
        # Objekt des Adam Optimizers mit Learnrate (Default 0.001)
        optimizer=Adam(learning_rate=0.0005),  #optimizer="Adam", # Übergabe als String (nur wenn in TF existent. Dabei werden immer Defaultwert genommen) oder alternativ als Funktionsaufruf:  tf.keras.optimizers.Adam
        metrics=["accuracy"] # (OPTIONAL) Wichtig: nicht die Funktion aufrufen, sondern das Funktionsobjekt übergben. Wichtig TF erwartet hier immer eine Liste
    )

    # Schritt 3: Training ausführen
    model.fit(
        x=x_train,
        y=y_train,
        epochs=10,                       # (OPTIONAL) Wieviele Epochen sollen trainiert werden. Default = 11
        batch_size=128,                  # (OPTIONAL) Number of samples per batch of computation. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of a dataset, generators, or keras.utils.Sequence instances (since they generate batches)
        verbose=1,                       # (OPTIONAL) Wenn Verbose =1, dann wird der Output jede Epoche ausgegeben
        validation_data=(x_test, y_test) # (OPTIONAL) Validation Daten
    )

    # Schritt 4 Testing (Optional). Die Evaluate Funktion returned einen Losswert und einen Metric-Wert
    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0      # (OPTIONAL) Wenn Verbose =1, dann wird der Output ausgegeben
    )
    print(f"Scores before saving: {scores}")


    # ...........
    # Speichern und Laden der Weights des Modells. Wichtig es muss in das Gleiche Modell geladen werden, damit die Weights wieder funktionieren. Spricch gleiche Anzahl an Neurone in und Outputs
    # Nach dem Speichern und Laden des Modells kann das Training und das Speichern auskommentiert werden
    # ...........
    model.save_weights(filepath=MODEL_FILE_PATH)  # Speichern der Weights als h5 Datei unter Fold er Models
    model.load_weights(filepath=MODEL_FILE_PATH)
    # Schritt 4.2 Testing (Optional). Die Evaluate Funktion returned einen Losswert und einen Metric-Wert
    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0  # (OPTIONAL) Wenn Verbose =1, dann wird der Output ausgegeben
    )
    print(f"Scores after loading: {scores}")


if __name__ == "__main__":
    nn_train()
