import os
import time
import random
import threading

import RPi.GPIO as GPIO

# Define sounds
SOUNDS = {11: {'audio': ['11.wav'], 'last_action': 0},
          12: {'audio': ['12.wav'], 'last_action': 0},
          13: {'audio': ['13.wav'], 'last_action': 0},
          15: {'audio': ['15.wav'], 'last_action': 0},
          16: {'audio': ['16.wav'], 'last_action': 0},
          18: {'audio': ['18.wav'], 'last_action': 0},
          22: {'audio': ['22.wav'], 'last_action': 0}
        }

# Ignore warning for now
GPIO.setwarnings(False) 

# Use physical pin numbering
GPIO.setmode(GPIO.BOARD) 

# Pin Setup
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Play sound using simple aplay command
def playSound(path):    
    print('PLAYING:', path)
    os.system('aplay ' + path)

# Button callback
def button_callback(channel):

    global SOUNDS

    # Time
    now = time.time()

    # Wait for timeout
    if now - SOUNDS[channel]['last_action'] > 2:
        
        # Save time of last button press
        SOUNDS[channel]['last_action'] = now

        # Randomly select a file to play
        afile = random.choice(SOUNDS[channel]['audio'])

        # Execute button actions        
        t = threading.Thread(target=playSound, args=('sounds/' + afile,))
        t.start()

# Register button events
GPIO.add_event_detect(11, GPIO.FALLING, callback=button_callback)
GPIO.add_event_detect(12, GPIO.FALLING, callback=button_callback)
GPIO.add_event_detect(13, GPIO.FALLING, callback=button_callback)
GPIO.add_event_detect(15, GPIO.FALLING, callback=button_callback)
GPIO.add_event_detect(16, GPIO.FALLING, callback=button_callback)
GPIO.add_event_detect(18, GPIO.FALLING, callback=button_callback)
GPIO.add_event_detect(22, GPIO.FALLING, callback=button_callback)

# Run until someone presses ctrl+c
while True:
    try:
        time.sleep(0.1)
    except KeyboardInterrupt:
        print('TERMINATED')
        break

# Clean up
print('CLEANING UP')
GPIO.cleanup()
    