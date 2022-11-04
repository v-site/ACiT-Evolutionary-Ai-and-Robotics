import cv2
import tempfile
import numpy as np
from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    import Image

vidcap = cv2.VideoCapture('../VideoResults/2022-10-31 11-58-54.mkv') # loads videofile
frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 # determin number of frames -1 as we want it zero-indexed
goodFrames = 0 # usable frames

with tempfile.TemporaryDirectory() as directory: # creates a temporary directory in AppData\Local\Temp

    print('Created temporary directory: %s' % directory, '\n') # prints the path to the created directory

    for _ in tqdm (range(frames), desc = 'Cleaning'): # runs for-loop with progress bar

        ret, frame = vidcap.read() #reads frame to memmory

        if not(ret) or np.average(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < 20: # checks if the frame exists and does not consist of mostly black (grayscale for computational simplicity)

            continue # skips to next loop iteration if bad frame found

        cv2.imwrite(directory + '/frame' + str(goodFrames) + '.jpg', frame) # stores good frames in temp directory
        goodFrames += 1 # counts stored frames

    print('\nDiscarded %d frames' % (frames - goodFrames), '\n') # prints amount of discarded bad frames

    background = Image.open(directory + '/frame0.jpg') # loads first frame as background for stacking

    for n in tqdm (range(goodFrames), desc = 'Stacking'): # runs for-loop with progress bar

        foreground = Image.open(directory + '/frame' + str(n) + '.jpg') # loads next frame as foreground for stacking

        new_img = Image.blend(background, foreground, 1 / goodFrames) # blends foreground and backround with a ratio of 1/goodFrames (makes all frames equally visible)
        new_img.save('../VideoResults/Stack.png', 'PNG') # stores the blended image in the results folder

        background = Image.open('../VideoResults/stack.png') # loads the blended image as background and repeats

background.show() # opens the final stacked image in native viewer
