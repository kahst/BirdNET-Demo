import os
import time
import json
import operator
from threading import Thread

import numpy as np
import pyaudio

from tensorflow import lite as tflite

from config import config as cfg
from utils import audio
from utils import image
from utils import log

DET = {}
FRAMES = np.array([], dtype='float32')
INTERPRETER = None
INPUT_LAYER_INDEX = -1
OUTPUT_LAYER_INDEX = -1

def openStream():       

    try:

        # Setup pyaudio
        paudio = pyaudio.PyAudio()
        
        # Stream Settings
        stream = paudio.open(format=pyaudio.paFloat32,
                            #input_device_index=0,
                            channels=1,
                            rate=cfg['SAMPLE_RATE'],
                            input=True,
                            frames_per_buffer=cfg['SAMPLE_RATE'] // 2)

        return stream

    except:
        return None

def record():

    global FRAMES

    # Open stream
    stream = openStream()
    
    while not cfg['KILL_ALL']:

        try:

            # Read from stream
            data = stream.read(cfg['SAMPLE_RATE'] // 2)
            data = np.fromstring(data, 'float32');
            FRAMES = np.concatenate((FRAMES, data))

            # Truncate frame count
            FRAMES = FRAMES[-int(cfg['SAMPLE_RATE'] * cfg['SPEC_LENGTH']):]

        except KeyboardInterrupt:
            cfg['KILL_ALL'] = True
            break
        except:
            FRAMES = np.array([], dtype='float32')
            stream = openStream()
            continue

def loadModel(model_file, config_file):

    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX

    log.p('LOADING TF LITE MODEL...', new_line=False)

    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()    

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

    # Load model-specific config
    cfg['LOAD'](config_file, ['CLASSES',
                              'SPEC_TYPE',
                              'MAGNITUDE_SCALE',
                              'WIN_LEN',
                              'SAMPLE_RATE',
                              'SPEC_FMIN',
                              'SPEC_FMAX',
                              'SPEC_LENGTH',
                              'INPUT_TYPE',
                              'INPUT_SHAPE'])

    log.p('DONE!')
    log.p(('INPUT LAYER INDEX:', INPUT_LAYER_INDEX))
    log.p(('OUTPUT LAYER INDEX:', OUTPUT_LAYER_INDEX))

    return interpreter    

def getSpeciesList():

    # Add selected species to white list
    cfg['WHITE_LIST'] = [# Species that have a sound file
                         'Sturnus vulgaris_European Starling',
                         'Delichon urbicum_Common House-Martin',
                         'Linaria cannabina_Eurasian Linnet',
                         'Ficedula hypoleuca_European Pied Flycatcher',
                         'Regulus regulus_Goldcrest',
                         'Emberiza citrinella_Yellowhammer',
                         'Cyanistes caeruleus_Eurasian Blue Tit',
                         'Phylloscopus collybita_Common Chiffchaff',
                         'Carduelis carduelis_European Goldfinch',
                         # Additional species
                         #'Parus major_Great Tit',
                         #'Passer domesticus_House Sparrow',
                         #'Erithacus rubecula_European Robin',
                         #'Phoenicurus ochruros_Black Redstart',
                         #'Fringilla coelebs_Common Chaffinch',
                         #'Turdus merula_Eurasian Blackbird'
                        ]

def getInput(sig):

    if cfg['INPUT_TYPE'] == 'raw':        

        # Prepare as input
        sample = audio.prepare(sig)

    else:        
        spec = audio.getSpec(sig,
                            rate=cfg['SAMPLE_RATE'],
                            fmin=cfg['SPEC_FMIN'],
                            fmax=cfg['SPEC_FMAX'],
                            win_len=cfg['WIN_LEN'],
                            spec_type=cfg['SPEC_TYPE'],
                            magnitude_scale=cfg['MAGNITUDE_SCALE'],
                            bandpass=True,
                            shape=(cfg['INPUT_SHAPE'][0], cfg['INPUT_SHAPE'][1]))

        # DEBUG: Save spec?
        if cfg['DEBUG_MODE']:
            image.saveSpec(spec, os.path.join(cfg['LOG_DIR'], 'spec.jpg'))

        # Prepare as input
        sample = image.prepare(spec)

    return sample

def flat_sigmoid(x, sensitivity=-1):
    return 1 / (1.0 + np.exp(sensitivity * x))

def predictionPooling(p, sensitivity=-1, mode='avg'):

    # Apply sigmoid function
    p = flat_sigmoid(p, sensitivity)

    # Mean exponential pooling for monophonic recordings
    if mode == 'mexp':
        p_pool = np.mean((p * 2.0) ** 2, axis=0)

    # Simple average pooling
    else:        
        p_pool = np.mean(p, axis=0)
    
    p_pool[p_pool > 1.0] = 1.0

    return p_pool

def predict(sample, interpreter):    

    # Make a prediction
    interpreter.set_tensor(INPUT_LAYER_INDEX, np.array(sample, dtype='float32'))
    interpreter.invoke()
    prediction = interpreter.get_tensor(OUTPUT_LAYER_INDEX)

    # Prediction pooling
    p_pool = predictionPooling(prediction, cfg['SENSITIVITY'])

    # Get label and scores for pooled predictions  
    p_labels = {}  
    for i in range(p_pool.shape[0]):
        label = cfg['CLASSES'][i]
        if cfg['CLASSES'][i] in cfg['WHITE_LIST']:
            p_labels[label] = p_pool[i]
        else:
            p_labels[label] = 0.0

    # Sort by score
    p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)
    
    return p_sorted

def analyzeStream(interpreter):

    # Time
    start = time.time()

    # Get signal from FRAMES
    sig = FRAMES.copy()

    # Do we have enough frames?
    if len(sig) < cfg['SAMPLE_RATE'] * cfg['SPEC_LENGTH']:
        return None

    # Prepare as input
    sample = getInput(sig) 

    # Make prediction
    p = predict(sample, interpreter)

    # Sort trough detections
    d = []
    for entry in p:

        # Store detections with confidence above threshold
        if entry[1] >= cfg['MIN_CONFIDENCE'] and p.index(entry) < 2:            

            # Save detection if it is a bird
            d.append({'species': entry[0], 'score': int(entry[1] * 100) / 100.0})

    return {'detections': d, 
            'audio': np.array(sig * 32767, dtype='int16'), 
            'timestamp': time.time(),
            'time_for_prediction': time.time() - start}

def save(p):

    # Time in UTC
    utc = time.strftime('%H:%M:%S', time.localtime(p['timestamp']))
    
    # Log    
    for detection in p['detections']:
        log.p((utc, int((p['time_for_prediction']) * 1000) / 1000.0), new_line=False)
        log.p((detection['species'], detection['score']), new_line=False)
        log.p('')

    # Save JSON response data
    data = {'prediction': {'0':{}}, 'time': p['time_for_prediction']}
    with open('stream_analysis.json', 'w') as jfile:

        for i in range(len(p['detections'])):
            label = p['detections'][i]['species']
            data['prediction']['0'][str(i)] = {'score': str(p['detections'][i]['score']), 'species': label}
            if i > 25:
                break

        json.dump(data, jfile)

def run():

    # Load model
    interpreter = loadModel(cfg['MODEL_PATH'], cfg['CONFIG_PATH'])

    # Load species list
    getSpeciesList()

    # Start recording
    log.p(('STARTING RECORDING WORKER'))
    recordWorker = Thread(target=record, args=())
    recordWorker.start()

    # Keep running...
    log.p(('STARTING ANALYSIS'))
    while not cfg['KILL_ALL']:

        try:

            # Make prediction
            p = analyzeStream(interpreter)

            # Save results
            if not p == None:
                save(p)

                # Sleep if we are too fast
                if 'time_for_prediction' in p:
                    if p['time_for_prediction'] < cfg['SPEC_LENGTH'] - cfg['SPEC_OVERLAP']:
                        time.sleep((cfg['SPEC_LENGTH'] - cfg['SPEC_OVERLAP']) - (p['time_for_prediction']))

            else:
                time.sleep(1.0)

        except KeyboardInterrupt:
            cfg['KILL_ALL'] = True
            break
        #except:
            #cfg.KILL_ALL = True

    # Done
    log.p(('TERMINATED'))


if __name__ == '__main__':    
    
    run()
    