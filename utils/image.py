import sys
sys.path.append("..")

import os
import numpy as np

from config import config as cfg

RANDOM = cfg['RANDOM']

def loadSpec(path):

    import cv2
    
    if path.rsplit('.', 1)[-1] == 'png':

        # Open spec as image
        spec = cv2.imread(path, 0)

        # Convert to floats between 0 and 1
        spec = np.asarray(spec / 255., dtype='float32')

    elif path.rsplit('.', 1)[-1] == 'npz':

        # Open raw spec data
        spec = np.load(path)['arr_0']

    return spec

def saveSpec(spec, filename):

    # Make dir
    if not os.path.exists(filename.rsplit(os.sep, 1)[0]):
        os.makedirs(filename.rsplit(os.sep, 1)[0])

    from PIL import Image
    im = Image.fromarray(spec * 255.0).convert("L")
    im.save(filename)

def normalize(spec):

    # Normalize
    if not spec.min() == 0 and not spec.max() == 0:
        spec -= spec.min()
        spec /= spec.max()
    else:
        spec = spec.clip(0, 1)

    return spec

def prepare(spec):
    
    # Add axis if 2D array
    if len(spec.shape) == 2:
        spec = spec[:, :, np.newaxis]

    # Add new dimension for batch size
    spec = np.expand_dims(spec, 0)
    return spec

def augment(spec):

    # Parse selected augmentations
    for aug in cfg['AUGMENTATIONS']:

        # Decide if we should apply this method
        if RANDOM.choice([True, False], p=[cfg['AUGMENTATION_PROBABILITY'], 1 - cfg['AUGMENTATION_PROBABILITY']]):

            # Apply augmentation
            if aug == 'v_roll':
                spec = v_roll(spec)
            elif aug == 'h_roll':
                spec = h_roll(spec)
            elif aug == 'v_stretch':
                spec = v_stretch(spec)
            elif aug == 'h_stretch':
                spec = h_stretch(spec)
            elif aug == 'noise':
                spec = noise(spec)
            elif aug == 'crop':
                spec = crop(spec)
            elif aug == 'v_flip':
                spec = v_flip(spec)
            elif aug == 'h_flip':
                spec = h_flip(spec)

    return spec

def resize(spec, width, height, interpolation='nearest'):
    
    import cv2

    if interpolation == 'nearest':
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_CUBIC

    # Squeeze resize: Resize image and ignore aspect ratio    
    return cv2.resize(spec, (width, height), interpolation=interpolation)

def v_flip(spec):

    import cv2
    
    return cv2.flip(spec, 0)

def h_flip(spec):

    import cv2
    
    return cv2.flip(spec, 1)

def crop(spec, top=0.1, left=0.1, bottom=0.1, right=0.1):

    h, w = spec.shape[:2]

    t_crop = max(1, int(h * RANDOM.uniform(0, top)))
    l_crop = max(1, int(w * RANDOM.uniform(0, left)))
    b_crop = max(1, int(h * RANDOM.uniform(0, bottom)))
    r_crop = max(1, int(w * RANDOM.uniform(0, right)))

    spec = spec[t_crop:-b_crop, l_crop:-r_crop]    
    spec = squeeze(spec, w, h)

    return spec

def v_stretch(spec, size=0.75, amount=0.25):

    # Only works for 2D images
    h, w = spec.shape

    # Random values
    a = int(h * RANDOM.uniform(0, amount))
    s = int(h * RANDOM.uniform(0, size))

    if a > 0 and s > 0:

        # Dummy image
        spec_s = np.zeros((h + a, w), dtype='float32')

        # Select region to scale
        v_start = RANDOM.randint(0, h - s)
        area = spec[v_start:v_start + s, :]
        area = resize(area, area.shape[1], area.shape[0] + a)

        # Fill dummy image
        spec_s[0:v_start, :] = spec[0:v_start, :]
        spec_s[v_start:v_start + area.shape[0], :] = area  
        spec_s[v_start + area.shape[0]:h + a, :] = spec[v_start + s:, :]

        # Resize and return
        return resize(spec_s, w, h)

    else:
        return spec

def h_stretch(spec, size=0.75, amount=0.25):

    # Only works for 2D images
    h, w = spec.shape

    # Random values
    a = int(w * RANDOM.uniform(0, amount))
    s = int(w * RANDOM.uniform(0, size))

    if a > 0 and s > 0:

        # Dummy image
        spec_s = np.zeros((h, w + a), dtype='float32')

        # Select region to scale
        h_start = RANDOM.randint(0, w - s)
        area = spec[:, h_start:h_start + s]
        area = resize(area, area.shape[1] + a, area.shape[0])

        # Fill dummy image
        spec_s[:, 0:h_start] = spec[:, 0:h_start]
        spec_s[:, h_start:h_start + area.shape[1]] = area  
        spec_s[:, h_start + area.shape[1]:w + a] = spec[:, h_start + s:]

        # Resize and return
        return resize(spec_s, w, h)

    else:
        return spec 

def h_roll(spec, amount=0.5):

    spec = np.roll(spec, int(spec.shape[1] * RANDOM.uniform(-amount, amount)), axis=1)

    return spec

def v_roll(spec, amount=0.05):

    spec = np.roll(spec, int(spec.shape[0] * RANDOM.uniform(-amount, amount)), axis=0)

    return spec

def noise(spec):

    import cv2

    # Choose one item from list of noise samples
    index = RANDOM.randint(len(cfg['NOISE_SAMPLES']))

    # Open and resize image
    spec2 = loadSpec(cfg['NOISE_SAMPLES'][index])
    spec2 = resize(spec2, spec.shape[1], spec.shape[0])

    # Generate random weights
    w1 = RANDOM.uniform(0.25, 0.75)
    w2 = 1 - w1

    # Add images using weights
    spec = cv2.addWeighted(spec, w1, spec2, w2, 0)

    # Normalize
    spec = normalize(spec)

    return spec

if __name__ == '__main__':

    spec = loadSpec('../example/spec.png')