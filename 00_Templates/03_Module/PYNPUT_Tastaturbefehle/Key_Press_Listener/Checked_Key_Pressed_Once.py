#https://chat.openai.com/c/480ba40b-3e6d-4825-832b-3abdee97bd18
#In diesem angepassten Skript verwenden wir ein Set namens pressed_keys,
# um die gedrückten Tasten zu speichern.
# Wenn eine Taste gedrückt wird (on_press), wird überprüft,
# ob sie bereits im Set pressed_keys ist.
# Wenn nicht, wird sie ausgegeben und zum Set hinzugefügt.
# Wenn die Taste losgelassen wird (on_release), wird sie aus dem Set entfernt.
# Dadurch wird sichergestellt, dass eine Taste nur einmalig ausgegeben wird, wenn sie gedrückt wird. Falls sie erneut gedrückt wird, wird sie nicht erneut ausgegeben.

from pynput.keyboard import Key, Listener

pressed_keys = set() # Ein Set ist in Python eine Datenstruktur, die eine Sammlung eindeutiger Elemente darstellt. Das bedeutet, dass ein Element in einem Set nicht mehrfach vorkommen kann. Wenn versucht wird, ein bereits vorhandenes Element hinzuzufügen, hat das keine Auswirkungen.

def on_press(key):
    if key not in pressed_keys:
        print('{0} pressed'.format(key)) # Der Platzhalter {0} gibt an, dass an dieser Stelle ein Wert eingefügt werden soll. Bei 2 Variablen würde {0} die erste und {1} die zweite heranziehen
        pressed_keys.add(key) # Flag setzen -> Key wird in set() angelegt.

def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key) # Flag löschen -> Key wird in set() angelegt.

    print('{0} release'.format(key))
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
while(1):
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()