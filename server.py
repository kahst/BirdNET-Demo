import os
import json
import time

import bottle
from bottle import route, run, request, static_file, response

import pyaudio

############################### URL ##################################

# STATIC PATH REQUESTS
@route('/static/:path#.+#', name='static')
def static(path):
    return static_file(path, root='static')

# INDEX
@route('/')
def root():
    return static_file('demo.html', root='./static/')

# AUDIO STREAM
@route('/stream')
def stream():

    CHUNK_SIZE = 1024
    SAMPLE_RATE = 44100

    # Setup pyaudio
    paudio = pyaudio.PyAudio()
     
    # Stream Settings
    stream = paudio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

    # Calculate datasize
    datasize = (CHUNK_SIZE * 1 * 16) // 8

    # Create wav header
    h = 'RIFF'.encode('ascii')
    h += to_bytes((datasize + 36), 4,'little')
    h += 'WAVE'.encode('ascii')

    # Create response header
    response.headers['Content-Type'] = 'audio/x-wav'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Connection'] = 'close'

    # Read from stream
    loops = 0
    while True:
        data = stream.read(CHUNK_SIZE)
        ch = chunkHeader(SAMPLE_RATE, 16, 1, datasize)
        if loops == 0:
            r = h + ch + data
            loops = 1
        else:
            r = ch + data

        yield r
        

def chunkHeader(sampleRate, bitsPerSample, channels, datasize):   

    #o += bytes("fmt ",'ascii')                                              # (4byte) Format Chunk Marker
    o = 'fmt '.encode('ascii')
    
    #o += (16).to_bytes(4,'little')                                          # (4byte) Length of above format data
    o += to_bytes(16, 4, 'little')

    #o += (1).to_bytes(2,'little')                                           # (2byte) Format type (1 - PCM)
    o += to_bytes(1, 2, 'little')

    #o += (channels).to_bytes(2,'little')                                    # (2byte)
    o += to_bytes(channels, 2, 'little')

    #o += (sampleRate).to_bytes(4,'little')                                  # (4byte)
    o += to_bytes(sampleRate, 4, 'little')

    #o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4,'little')  # (4byte)
    o += to_bytes(sampleRate * channels * bitsPerSample // 8, 4, 'little')

    #o += (channels * bitsPerSample // 8).to_bytes(2,'little')               # (2byte)
    o += to_bytes(channels * bitsPerSample // 8, 2, 'little')

    #o += (bitsPerSample).to_bytes(2,'little')                               # (2byte)
    o += to_bytes(bitsPerSample, 2, 'little')

    #o += bytes("data",'ascii')                                              # (4byte) Data Chunk Marker
    o += 'data'.encode('ascii')

    #o += (datasize).to_bytes(4,'little')                                    # (4byte) Data size in bytes
    o += to_bytes(datasize, 4, 'little')

    return o

def to_bytes(n, length, endianess='big'):
    h = '%x' % n
    s = ('0'*(len(h) % 2) + h).zfill(length*2).decode('hex')
    return s if endianess == 'big' else s[::-1]       

# JSON ACTION
@route('/process', method='POST')
def process():
    ip = request.environ.get('REMOTE_ADDR')
    data = json.loads(request.forms.get('json'))
    data['ip'] = ip    
    return execute(data)

# REQUEST RESPONSE
def make_response(data, success=True):

    try:
        if success:
            data['response'] = 'success'
        else:
            data['response'] = 'error'
        return json.dumps(data, ensure_ascii=False, encoding='iso-8859-1')
    except:
        log.e(('Response Error!', data), discard=True)
        data = {}
        data['response'] = 'error'
        return json.dumps(data)

# EXECUTE ACTION
def execute(json_data):

    # Init
    data = {}

    # Parse Array
    action = json_data['action']
    ip = str(json_data['ip'])

    # Status
    print(ip, '|', 'action =', action)

    ##########  TEST NEURAL NET  #############
    try:

        if action == 'analysis':

            with open('stream_analysis.json', 'r') as jfile:
                data = json.load(jfile)
          
    except:
        data['response'] = 'error'

    return make_response(data)

############################## SERVER ################################
if __name__ == '__main__':

    # RUN SERVER
    print('STREAM SERVER UP AND RUNNING!')
    run(host='localhost', port=8080)

