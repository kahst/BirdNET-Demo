import os
import json
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Base paths
BASE_PATH = ''

# Random seed for initialization of all generators
RANDOM_SEED = 1337

# Save config as json file
def save(path):

    # Remove non-serializable entries
    del config['RANDOM']
    del config['SAVE']
    del config['LOAD']

    with open(path, 'w') as cfile:
        json.dump(config, cfile)

# Load config from json file
def load(path, entries=[]):

    global config

    with open(path, 'r') as cfile:
        c = json.load(cfile)
    
    # Overwrite selected or missing entries
    for entry in c:
        if not entry in config or entry in entries:
            config[entry] = c[entry]

# Config dict
config = {

    # Random generator
    'RANDOM_SEED': RANDOM_SEED,
    'RANDOM': np.random.RandomState(RANDOM_SEED),

    # Spectrogram properties
    'SPEC_SHAPE': (64, 384), # (height, width)
    'SPEC_TYPE': 'melspec',
    'MAGNITUDE_SCALE': 'nonlinear',
    'WIN_LEN': 512,
    'SAMPLE_RATE': 48000,
    'SPEC_FMIN': 150,
    'SPEC_FMAX': 15000,

    # Settings for spectrogram extraction
    'SPEC_LENGTH': 3.0,
    'SPEC_OVERLAP': 2.5,
    'SPEC_MINLEN': 1.0,

    # Model hyperperams
    'INPUT_TYPE': 'spec',

    # Soundscape analysis
    'MODEL_PATH': os.path.join(BASE_PATH, 'model', 'BirdNET_1000_TF_ResNet_50K_model.tflite'),
    'CONFIG_PATH': os.path.join(BASE_PATH, 'model', 'BirdNET_1000_TF_ResNet_50K_config.json'),
    'SPECS_PER_PREDICTION': 1,
    'SENSITIVITY': -0.85,
    'MIN_CONFIDENCE': 0.05,
    'DETECTION_COOLDOWN': 2.0,
    'MIN_DETECTION_COUNT': 2,

    # Logging
    'LOG_DIR': os.path.join(BASE_PATH, 'logs'),

    # Functions
    'SAVE': save,
    'LOAD': load,

    # Globals
    'KILL_ALL': False,
    'DEBUG_MODE': True,

}
