import time
import operator
from threading import Thread

import numpy as np
import pyaudio

from tensorflow import lite as tflite

from save_detections import save_detections
from save_audio import save_audio, save_audio_from_detection

from config import config as cfg
from metadata import grid
from utils import audio
from utils import image
from utils import log

DET = {}
FRAMES = np.array([], dtype='float32')
RAW_AUDIO = np.array([], dtype='float32')
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
    global RAW_AUDIO

    # Open stream
    stream = openStream()
    
    while not cfg['KILL_ALL']:

        try:

            # Read from stream
            data = stream.read(cfg['SAMPLE_RATE'] // 2)
            data = np.fromstring(data, 'float32');
            FRAMES = np.concatenate((FRAMES, data))
            RAW_AUDIO = np.concatenate((RAW_AUDIO, data))

            # Truncate frame count
            FRAMES = FRAMES[-int(cfg['SAMPLE_RATE'] * cfg['SPEC_LENGTH']):]
            RAW_AUDIO = RAW_AUDIO[-int(cfg['SAMPLE_RATE'] * cfg['RAW_AUDIO_LENGTH']):]

        except KeyboardInterrupt:
            cfg['KILL_ALL'] = True
            break
        except:
            FRAMES = np.array([], dtype='float32')
            RAW_AUDIO = np.array([], dtype='float32')
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

def loadGridData():

    # Load eBird data
    grid.load()
    
    # Set species white list
    getSpeciesList()        

def getSpeciesList():

    # Get current week
    week = grid.getWeek()

    # Determine species lists (only year round is supported)
    if cfg['USE_EBIRD_CHECKLIST']:
        cfg['WHITE_LIST'], cfg['BLACK_LIST'] = grid.getSpeciesLists(cfg['DEPLOYMENT_LOCATION'][0], cfg['DEPLOYMENT_LOCATION'][1], week, cfg['EBIRD_FREQ_THRESHOLD'])
    else:
        cfg['WHITE_LIST'] = cfg['CLASSES']

    #for s in sorted(cfg['WHITE_LIST']):
    #    log.p(s)

    log.p(('GPS LOCATION:', cfg['DEPLOYMENT_LOCATION']))
    log.p(('SPECIES:', len(cfg['WHITE_LIST'])), new_line=True)    

    # Add human to white list
    cfg['WHITE_LIST'].append('Human_Human')

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
            image.saveSpec(spec, 'log/spec.jpg')

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

        # Avoid human detections if confidence > 3%
        if entry[0] in ['Human_Human'] and entry[1] >= 0.03:
            d = []
            cfg['LAST_HUMAN_DETECTION'] = time.time()
            break

        # Store detections with confidence above threshold
        elif entry[1] >= cfg['MIN_CONFIDENCE'] and p.index(entry) < 1:            

            # Save detection if it is a bird
            d.append({'species': grid.getSpeciesCode(entry[0]), 'score': int(entry[1] * 100) / 100.0})

    return {'detections': d, 
            'audio': np.array(sig * 32767, dtype='int16'), 
            'timestamp': time.time(),
            'time_for_prediction': time.time() - start}

def save(p):

    # Time in UTC
    utc = time.strftime('%H:%M:%S', time.localtime(p['timestamp']))
    
    # Log
    log.p((utc, int((p['time_for_prediction']) * 1000) / 1000.0), new_line=False)
    for detection in p['detections']:
        log.p((detection['species'], detection['score']), new_line=False)
    log.p('')

    # Count and validate detections
    validated_detections = validateAndCount(p)

    # Send to save_detections.py
    for d in validated_detections:
        save_detections(d)

        if cfg['DEBUG_MODE']:
            save_audio_from_detection(d)

def clearDetection():

    return {'score': 0, 'timestamp': 0, 'time_for_prediction': 0, 'audio': [], 'count': 0, 'cooldown': cfg['DETECTION_COOLDOWN']}

def validateAndCount(p):

    global DET

    # Add current predictions and count
    for d in p['detections']:

        # Allocate dict
        if not d['species'] in DET:
            DET[d['species']] = clearDetection()

        # New best score?
        if d['score'] > DET[d['species']]['score']:

            # Store detection data
            DET[d['species']]['score'] = d['score']
            DET[d['species']]['timestamp'] = p['timestamp']
            DET[d['species']]['time_for_prediction'] = p['time_for_prediction']
            DET[d['species']]['audio'] = p['audio']

        # Count
        DET[d['species']]['count'] += 1

    # Reduce cooldowns and validate count
    p, r = [], []
    for species in DET:
        DET[species]['cooldown'] -= 1.0
        if DET[species]['cooldown'] < 0.0:
            if DET[species]['count'] >= cfg['MIN_DETECTION_COUNT'] and DET[species]['score'] > 0.15:
                p.append({'detections': [{'score': DET[species]['score'], 'species': species}],
                          'timestamp': DET[species]['timestamp'],
                          'time_for_prediction': DET[species]['time_for_prediction'],
                          'audio': DET[species]['audio']})
                log.p(('>>>  SAVING:', species, '  <<<'))
            r.append(species)

    # Remove species with expired cooldown
    for species in r:
        del DET[species]

    return p

def run():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat', type=float, default=-1, help='Recording location latitude. Set -1 to ignore.')
    parser.add_argument('--lon', type=float, default=-1, help='Recording location longitude. Set -1 to ignore.')

    args = parser.parse_args()

    if not args.lat == -1 and not args.lon == -1:
        cfg['FORCE_GPS'](args.lat, args.lon)

    # Load model
    interpreter = loadModel(cfg['MODEL_PATH'], cfg['CONFIG_PATH'])

    # Update GPS location if possible
    cfg['UPDATE_GPS']()

    # Load eBird grid data
    loadGridData()

    # Start recording
    log.p(('STARTING RECORDING WORKER'))
    recordWorker = Thread(target=record, args=())
    recordWorker.start()

    # Keep running...
    log.p(('STARTING ANALYSIS'))
    while not cfg['KILL_ALL']:

        try:

            # Save raw audio?
            if time.time() - cfg['LAST_RAW_AUDIO_SAVE'] > cfg['RAW_AUDIO_SAVE_INTERVAL'] and len(RAW_AUDIO) >= cfg['SAMPLE_RATE'] * cfg['RAW_AUDIO_LENGTH'] and time.time() - cfg['LAST_HUMAN_DETECTION'] > cfg['RAW_AUDIO_LENGTH']:
                log.p(('>>>  SAVING RAW AUDIO  <<<'))
                save_audio(RAW_AUDIO, prefix='raw_')
                cfg['LAST_RAW_AUDIO_SAVE'] = time.time()

                # We also want to update the species list occasionally                
                getSpeciesList()

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
    