# Tensorboard
# Basierend auf File aus Chapter5_4_MINSTClassification
# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/8533352#overview

import os

import numpy as np
from tensorflow.keras.callbacks import TensorBoard  # NEU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import TruncatedNormal  # Initialisierung für Gewichte
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam  # Optimizer
from tensorflow.keras.utils import to_categorical  # Für OneHot Vektor

# https://www.tensorflow.org/api_docs/python/tf/keras
# https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/learn/lecture/17300996#overview

MODELS_DIR = os.path.abspath(r"M:\Google Drive\Programming\Python\01_Tutorials\Udemy Kurs Deep Learning\models")  # Speicherort für Modelle
MODELS_FILE_PATH = os.path.join(MODELS_DIR, "mnist_model.h5")  # h5 Datei wird von Keras zum speichern der Gewichte verwendet

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

LOGS_DIR = os.path.abspath(r"M:\Google Drive\Programming\Python\01_Tutorials\Udemy Kurs Deep Learning\logs")
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_model_3")  # Muss einzigartiger name sein

# Umformatierung unserer In-/Output Daten aus MNIST Dataset
def prepare_Dataset(num_features: int, num_targets: int):

    # Test Train Split
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Lädt Datensatz aus Inet herunter

    # Umformatierung der Datentypen nach Float32
    x_train = x_train.reshape(-1, num_features).astype(np.float32) # Zwingt das Format in eine Spalte  -> Keras erwartet Shape.(X, 1)
    x_test = x_test.reshape(-1, num_features).astype(np.float32) # Zwingt das Format in eine Spalte  -> Keras erwartet Shape.(X, 1)

    y_train = to_categorical(y_train, num_classes=num_targets, dtype=np.float32)  # ONEHOT Vektor. Klassifikation kommen duetlich besser klar wenn die y-Werte als OneHot Vektor vorliegen
    y_test = to_categorical(y_test, num_classes=num_targets, dtype=np.float32) # ONEHOT Vektor

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test) # Doppeltes Tupel

# Model mit bestimmter Anzahl an Layern erzeugen -> Returned wird das Sequential Objekt
def build_model(num_features: int, num_targets: int) -> Sequential:  # Inpput: num_features ; Output num_targets

    init_w = TruncatedNormal(mean=0.0, stddev=0.01) # Startwert für Gewicht deklarieeren. Zufallsverteilung um 0 mit Standardabweichung 0.01
    init_b = Constant(value=0.0) # Initialisierung bias Neuronen. INFO: Per Default werden bias Neuronen immer mit 0 in keras initialisiert

    # Der Output des ersten Layers ist der Input des folgenden Layers
    model = Sequential() # Objekt der Sequential Klasse erstellen
    # Beim ersten Layer MUSS Input Shape deklariert werden
    model.add(Dense(units=500, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,))) # Layer hinzufügen -> Erinnerung  Dense Layer: Hidden Layer mit Input Layer Verbinden. Units -> Wieviele Neuronen hat der Hidden Layer, input_shape -> Shape der Eingangsfeatures
    model.add(Activation("relu")) # Aktivierungsfunktion verknüpfen

    # Hidden Layer 2
    model.add(Dense(units=250, kernel_initializer=init_w, bias_initializer=init_b))  # Layer hinzufügen -> Erinnerung  Dense Layer: Hidden Layer mit Input Layer Verbinden. Units -> Wieviele Neuronen hat der Hidden Layer, input_shape -> Shape der Eingangsfeatures
    model.add(Activation("relu"))  # Aktivierungsfunktion verknüpfen

    # Hidden Layer 3
    model.add(Dense(units=50, kernel_initializer=init_w, bias_initializer=init_b))  # Layer hinzufügen -> Erinnerung  Dense Layer: Hidden Layer mit Input Layer Verbinden. Units -> Wieviele Neuronen hat der Hidden Layer, input_shape -> Shape der Eingangsfeatures
    model.add(Activation("relu"))  # Aktivierungsfunktion verknüpfen


    model.add(Dense(units=num_targets, kernel_initializer=init_w, bias_initializer=init_b)) # Output Layer verknüppfen
    model.add(Activation("softmax"))  # WICHTIG bei Klassifikation immer SOFTMAX-Aktivierungsfunktion -> Outputwerte werden als Wahrscheinlichkeiten ausgegeben
    # WICHTIG bei Regression keine SOFTMAX am Ende
    model.summary() # Ausgabe der Struktur des Netzwerkes. Erinnerung Activation Funktionen haben 0 Parameter. Die Anzahl der Parameter ergibt sich aus (Anz Input* Anz Neuronen)+ Anz Bias Einträge

    return model

if __name__ == "__main__":

    num_features = 784  # Anzahl der Features kommt aus Dataset: 60000 Bilder a 28x28 Pixel = 784Pixel (Muss für Dense Funktion als Vektor vorliegen)
    num_targets = 10  # Anzahl der Klassen (Wird auch für ONEHOT Vektor verwendet)

    (x_train, y_train), (x_test, y_test) = prepare_Dataset(num_features, num_targets)  # Funktionsaufruf für Dataset

    # Schritt 1:  Modell erstellen
    model = build_model(num_features, num_targets)

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
        epochs=20,                        # (OPTIONAL) Wieviele Epochen sollen trainiert werden. Default = 11
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
