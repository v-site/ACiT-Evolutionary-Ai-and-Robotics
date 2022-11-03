import cv2
import tempfile
import numpy as np
from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    import Image

vidcap = cv2.VideoCapture('VideoResults/2021-05-06 15-39-21.mkv')
frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
f = 0

with tempfile.TemporaryDirectory() as directory:

    print('Created temporary directory: \n %s' % directory, '\n')

    for _ in tqdm (range(frames), desc='Cleaning'):

        ret,frame = vidcap.read()

        if not(ret) or np.average(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < 20:
            continue

        cv2.imwrite(directory + '/frame' + str(f) + '.jpg', frame)
        f+=1

    print('\n', 'Discarded %d frames' % (frames-f), '\n')

    background = Image.open(directory + '/frame0.jpg')

    for n in tqdm (range(f), desc='Stacking'):

        overlay = Image.open(directory + '/frame' + str(n) + '.jpg')

        new_img = Image.blend(background, overlay, 1/f)
        new_img.save('VideoResults/Stack.png','PNG')

        background = Image.open('VideoResults/Stack.png')

background.show()
