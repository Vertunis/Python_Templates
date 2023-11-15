import sys, bot, keyboard
from time import sleep

try:
	BOT = bot.Bot()
	toggle_button = '.'
	#toggle_button = '^'
	enabled = False
	last_state = False
	first_time = True

	print('AFK script started, press "%s" key to start/stop bot.' % toggle_button)

	while True:		
		key_down = keyboard.is_pressed(toggle_button) # Prüft ob Toggle Button gedrückt -> Returns True wenn gedrückt
		if key_down != last_state: # Wenn erstmalig gedrückt, dann !=
			last_state = key_down  # Übernimmt aktuellen Zustand. Wenn erstmalig toggle button gedrücked dann TRUE
			if last_state:
				enabled = not enabled
				if enabled and first_time:
					print("\nStarting bot...")
					BOT.start()
					first_time = False
				elif enabled:
					BOT.resume()
				else:
					BOT.pause()
except SystemExit:
	pass
except KeyboardInterrupt:
	sys.exit()