import tempfile
import numpy as np
import cv2

from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    import Image


vidcap = cv2.VideoCapture('VideoResults/CAtestGene.mkv')
frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
f = 0

with tempfile.TemporaryDirectory() as directory:

    print('Created temporary directory: %s' % directory)

    ret,frame = vidcap.read()
    f+=1

    while np.average(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < 20:
        ret,frame = vidcap.read()
        f+=1

    cv2.imwrite(directory + '/frame.jpg', frame)
    background = Image.open(directory + '/frame.jpg')

    for _ in tqdm (range(frames-f), desc='Stacking'):

        if f < frames:

            ret,frame = vidcap.read()
            f+=1

            if np.average(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 20:

                cv2.imwrite(directory + '/frame.jpg', frame)
                overlay = Image.open(directory + '/frame.jpg')

                new_img = Image.blend(background, overlay, 1/frames)
                new_img.save('VideoResults/Stack.png','PNG')

                background = Image.open('VideoResults/Stack.png')

background.show()
