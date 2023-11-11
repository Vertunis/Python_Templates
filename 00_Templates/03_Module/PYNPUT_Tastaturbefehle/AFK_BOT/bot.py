import threading, time, random

from pynput.keyboard import Key, Controller


def PressKey(key, controller):
	print("[BOT]: Pressing %s" % key)
	controller.press(key)
	time.sleep(random.uniform(1, 2))   # Warte zwischen 1 und 2 Sekunden
	print("[BOT]: Releasing %s" % key)
	controller.release(key)


class Bot(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.paused = False
		self.pause_condition = threading.Condition(threading.Lock())
		self.keys = ["w","a","s","d"]
		self.special_keys = [Key.esc, Key.backspace, Key.space, Key.ctrl_l, Key.shift_l] # Sonderzeichen müssel mit Key.<Namen> Bezeichnet werden
		self.keyboard = Controller()

	def run(self):
		while True:
			with self.pause_condition:
				while self.paused:
					self.pause_condition.wait()

				time.sleep(random.uniform(1, 2))   # Warte zwischen 1 und 2 Sekunden
				#PressKey(self.keys[0], self.keyboard)
				#PressKey(self.keys[2], self.keyboard)

				#self.keyboard.press(Key.shift_l) # Manuell Sonderzeichen pressen
				#self.keyboard.release(Key.shift_l)

				PressKey(self.special_keys[4], self.keyboard)
			time.sleep(5)

	def pause(self):
		print("\n[BOT]: Pausing...")
		self.paused = True
		self.pause_condition.acquire()

	def resume(self):
		print("[BOT]: Resuming...\n")
		self.paused = False
		self.pause_condition.notify()
		self.pause_condition.release()