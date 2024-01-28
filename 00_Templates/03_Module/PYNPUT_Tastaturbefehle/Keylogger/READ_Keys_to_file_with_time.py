from pynput.keyboard import Key, Listener
import time
from datetime import datetime

# Funktion zum Formatieren des Datums und der Uhrzeit
def get_formatted_datetime():
    now = datetime.now()
    return now.strftime('%Y%m%d_%H%M%S')

# Dateipfad für die Ausgabedatei
output_file_path = get_formatted_datetime() + '_tastenanschlaege.txt'

# Initialisiere das Zeitstempel-Dictionary
timestamps = {}

def on_press(key):
    # Speichere den Startzeitpunkt des Tastenanschlags
    timestamps[key] = time.time()

def on_release(key):
    with open(output_file_path, 'a') as f:
        if key in timestamps:
            # Berechne die Dauer des Tastenanschlags
            duration = time.time() - timestamps[key]
            #f.write('{0} pressed for {1:.2f} seconds\n'.format(key, duration))
            f.write('{0};{1:.2f}\n'.format(key, duration))
            del timestamps[key]  # Entferne den Startzeitpunkt der Taste

        f.write('{0} release\n'.format(key))

        if key == Key.esc:
            # Stop listener
            return False

# Collect events until released
with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
