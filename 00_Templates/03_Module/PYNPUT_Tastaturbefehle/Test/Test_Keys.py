import threading
import time
import random
import ctypes

from pynput.keyboard import Key, Controller

keys = ["w", "a", "s", "d"]

def PressKey(key, controller):
    print("[BOT]: Pressing %s" % key)
    controller.press(key)
    time.sleep(2)
    controller.release(key)
    print("[BOT]: Releasing %s" % key)

def ToggleNumLock(status):

    if status != 0:
        ctypes.windll.user32.keybd_event(0x90, 0, 0x0002, 0)  # Key down
    else:
        ctypes.windll.user32.keybd_event(0x90, 0, 0x0002 | 0x0004, 0)  # Key up

if __name__ == "__main__":


    keyboard = Controller()

    ToggleNumLock(1) # Numlock drücken
    ToggleNumLock(0) # Numlock loslassen

    PressKey(keys[0], keyboard)
    PressKey(keys[2], keyboard)

    time.sleep(1)

    keyboard.press(Key.backspace) # Buchstaben löschen