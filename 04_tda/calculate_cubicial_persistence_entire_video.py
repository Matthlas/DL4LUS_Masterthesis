import argparse
import os
import numpy as np
import cv2
import numpy as np
from PIL import Image
from gtda.homology import CubicalPersistence

import os
import pandas as pd

import sys; sys.path.insert(0, "../utils/")
from data_utils import get_data_location

DATA_PATH = get_data_location()

# Parse the arguments
parser = argparse.ArgumentParser()
# Path to the post-processed segmentation masks (npy files, shape: (n_frames, X, Y), should only contain int values 0-4 representing the different segmentations)
parser.add_argument("--video_path", type=str, default="/itet-stor/mrichte/covlus_bmicnas02/cropped_videos/mp4", help="Path to the video files")
# Number of the file to process
parser.add_argument("--file_no", type=int, help="Number of the file to process")
# Scale factor for the images
parser.add_argument("--scale_factor", type=float, default=5, help="Scale factor for the images")
# Save path
parser.add_argument("--save_path", type=str, default="/itet-stor/mrichte/covlus_bmicnas02/tda_entire_vid", help="Path to save the results")
# Exclude already processed files
parser.add_argument("--exclude_processed", type=bool, default=True, help="Exclude already processed files")
args = parser.parse_args()

# Path to the video path
VIDEO_PATH = args.video_path
# Number of the file to process
FILE_NO = args.file_no
# Scale factor for the images
SCALE_FACTOR = args.scale_factor
# Save path
SAVE_PATH = args.save_path
# Exclude already processed files
EXCLUDE_PROCESSED = args.exclude_processed

############ VIDEO PATH ############
# List all segmentation files 
vid_files = os.listdir(VIDEO_PATH)
# keep only files that end with .npy
vid_files = [x for x in vid_files if x.endswith(".mp4")]

# Check if FILE_NO is valid
if FILE_NO >= len(vid_files):
    raise ValueError("FILE_NO is too large")

vid = os.path.join(VIDEO_PATH, vid_files[FILE_NO])

############ SAVE PATH ############
# Get the name of the video file
vid_name = vid_files[FILE_NO].split(".")[0]
# Create the save path
save_file = os.path.join(SAVE_PATH, vid_name + ".pickle")
# Check if the file already exists
if EXCLUDE_PROCESSED and os.path.exists(save_file):
    print("File already exists, skipping")
    sys.exit()

############ LOAD VIDEO ############
print("Loading video")
capture = cv2.VideoCapture(vid)
video_arr = []
video_ind = 0
while True:
    # capture frame-by-frame from video file
    ret, frame = capture.read()
    if not ret:
        break
    frame = Image.fromarray(frame, mode="RGB")
    frame = frame.convert('L')
    frame = np.asarray(frame)
    video_ind += 1
    video_arr.append(frame)
video_arr = np.array(video_arr)
capture.release()

############ PERSISTENCE DIAGRAMS ############
print("Calculating persistence diagrams")

CP = CubicalPersistence(homology_dimensions=(0,1,2), n_jobs=-1)
cubical_persistence = CP.fit_transform(video_arr[np.newaxis, :, :, :])

# Save the persistence diagrams
print("Saving the persistence diagrams")
import pickle
with open(save_file, "wb") as f:
    pickle.dump(cubical_persistence, f)

print("Done")