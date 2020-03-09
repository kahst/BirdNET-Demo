import os
import time
import random
import threading

import RPi.GPIO as GPIO

# Define sounds
SOUNDS = {11: {'audio': ['CS_01.wav', 'CS_02.wav', 'CS_06.wav', 'CS_07.wav', 'CS_08.wav', 'CS_09.wav'], 'last_action': 0},
          12: {'audio': ['CHM_01.wav', 'CHM_02.wav', 'CHM_03.wav', 'CHM_04.wav', 'CHM_05.wav'], 'last_action': 0},
          13: {'audio': ['CL_01.wav', 'CL_02.wav', 'CL_03.wav', 'CL_05.wav', 'CL_06.wav'], 'last_action': 0},
          15: {'audio': ['EPF_01.wav', 'EPF_02.wav', 'EPF_03.wav', 'EPF_04.wav', 'EPF_05.wav'], 'last_action': 0},
          16: {'audio': ['CCC_01.wav', 'EBT_01.wav', 'EGF_01.wav', 'GC_01.wav', 'YH_01.wav'], 'last_action': 0},
          18: {'audio': [], 'last_action': 0},
          22: {'audio': [], 'last_action': 0}
        }

# Define random playback interval
LAST_RANDOM_PLAYBACK = time.time()

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

# Choose random sound to play
def playRandomSound():

    global LAST_RANDOM_PLAYBACK

    # Time
    now = time.time()

    # Check if recent button action
    for c in SOUNDS:
        if now - SOUNDS[c]['last_action'] < 20:
            return None

    # Check for last random playback
    if not now - LAST_RANDOM_PLAYBACK < random.randint(10, 30):
        LAST_RANDOM_PLAYBACK = now
        button_callback(random.choice([11, 12, 13, 15, 16]), True)

# Button callback
def button_callback(channel, isRandom=False):

    global SOUNDS

    # Time
    now = time.time()

    # Wait for timeout
    if now - SOUNDS[channel]['last_action'] > 1.5:
        
        # Save time of last button press
        if not isRandom:
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
        playRandomSound()
    except KeyboardInterrupt:
        print('TERMINATED')
        break
    except:
        continue

# Clean up
print('CLEANING UP')
GPIO.cleanup()
    