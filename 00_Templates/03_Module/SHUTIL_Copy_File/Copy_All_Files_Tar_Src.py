import os
import shutil

def kopiere_dateien_von_quellordner_nach_zielordner(quellordner, zielordner):
    # Überprüfe, ob der Zielordner existiert, andernfalls erstelle ihn
    if not os.path.exists(zielordner):
        os.makedirs(zielordner)

    # Durchlaufe alle Dateien im Quellordner
    for dateiname in os.listdir(quellordner):
        quelle_pfad = os.path.join(quellordner, dateiname)
        ziel_pfad = os.path.join(zielordner, dateiname)

        # Kopiere die Datei in den Zielordner
        shutil.copy2(quelle_pfad, ziel_pfad)
        print(f"{dateiname} kopiert")

if __name__ == "__main__":
    # Lies die Quell- und Zielordner aus einer Textdatei aus
    with open("pfad.txt", "r") as file:
        pfad_text = file.read().splitlines()

    if len(pfad_text) == 2:
        quellordner, zielordner = pfad_text
        kopiere_dateien_von_quellordner_nach_zielordner(quellordner, zielordner)
    else:
        print("Die Textdatei sollte genau zwei Zeilen haben: Quellpfad und Zielpfad.")
