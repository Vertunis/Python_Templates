# https://stackoverflow.com/questions/24072790/how-to-detect-key-presses

from pynput.keyboard import Key, Listener

def on_press(key):
    print('{0} pressed'.format(key)) # Der Platzhalter {0} gibt an, dass an dieser Stelle ein Wert eingefügt werden soll. Bei 2 Variablen würde {0} die erste und {1} die zweite heranziehen

def on_release(key):
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