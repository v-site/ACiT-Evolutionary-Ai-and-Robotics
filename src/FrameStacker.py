import cv2
import tempfile
import numpy as np
from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    import Image

# loads videofile
vidcap = cv2.VideoCapture('../VideoResults/2022-10-31 11-58-54.mkv') 

# determin number of frames -1 as we want it zero-indexed
frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 

# usable frames
goodFrames = 0

# creates a temporary directory in AppData\Local\Temp
with tempfile.TemporaryDirectory() as directory:

    # prints the path to the created directory
    print('Created temporary directory: %s' % directory, '\n') 
    
    # runs for-loop with progress bar
    for _ in tqdm (range(frames), desc = 'Cleaning'):

        #reads frame to memmory
        ret, frame = vidcap.read() 
        
        # checks if the frame exists and does not consist of mostly black (grayscale for computational simplicity)
        if not(ret) or np.average(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < 20: 

            # skips to next loop iteration if bad frame found
            continue 

        # stores good frames in temp directory
        cv2.imwrite(directory + '/frame' + str(goodFrames) + '.jpg', frame)
        
        # counts stored frames
        goodFrames += 1 

    # prints amount of discarded bad frames
    print('\nDiscarded %d frames' % (frames - goodFrames), '\n') 

    # loads first frame as background for stacking
    background = Image.open(directory + '/frame0.jpg') 

    # runs for-loop with progress bar
    for n in tqdm (range(goodFrames), desc = 'Stacking'): 
        
        # loads next frame as foreground for stacking
        foreground = Image.open(directory + '/frame' + str(n) + '.jpg') 

        # blends foreground and backround with a ratio of 1/goodFrames (makes all frames equally visible)
        new_img = Image.blend(background, foreground, 1 / goodFrames)
        
        # stores the blended image in the results folder
        new_img.save('../VideoResults/Stack.png', 'PNG') 

        # loads the blended image as background and repeats
        background = Image.open('../VideoResults/stack.png')

# opens the final stacked image in native viewer
background.show() 
