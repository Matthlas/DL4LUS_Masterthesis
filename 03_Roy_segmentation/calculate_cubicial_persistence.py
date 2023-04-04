import argparse
import os
import numpy as np
import time
import pandas as pd
from skimage import measure # For marching cubes
# from gtda.homology import VietorisRipsPersistence # For persistence diagrams
from gtda.homology import CubicalPersistence
from tda_utils import load_gif, filter_frame, trim_zeros # Own functions

# Parse the arguments
parser = argparse.ArgumentParser()
# Path to the post-processed segmentation masks (npy files, shape: (n_frames, X, Y), should only contain int values 0-4 representing the different segmentations)
parser.add_argument("--segmentation_path", type=str, default="/itet-stor/mrichte/covlus_bmicnas02/cropped_videos_segmented/post_processing", help="Path to the segmentation masks")
# Path to the processed gif file containing the raw video to be masked
parser.add_argument("--gif_path", type=str, default="/itet-stor/mrichte/covlus_bmicnas02/cropped_videos_segmented", help="Path to the processed gif file")
# Number of the file to process
parser.add_argument("--file_no", type=int, help="Number of the file to process")
# Scale factor for the images
parser.add_argument("--scale_factor", type=float, default=5, help="Scale factor for the images")
# Save path
parser.add_argument("--save_path", type=str, default="/itet-stor/mrichte/covlus_bmicnas02/cropped_videos_segmented/cubicial_persistence", help="Path to save the results")
# Exclude already processed files
parser.add_argument("--exclude_processed", type=bool, default=True, help="Exclude already processed files")
args = parser.parse_args()

# Path to the segmentation masks
SEGMENTATION_PATH = args.segmentation_path
# Path to the processed gif file containing the raw video to be masked
GIF_PATH = args.gif_path
# Number of the file to process
FILE_NO = args.file_no
# Scale factor for the images
SCALE_FACTOR = args.scale_factor
# Save path
SAVE_PATH = args.save_path
# Exclude already processed files
EXCLUDE_PROCESSED = args.exclude_processed

############ SEGMENTATION MASKS ############
# List all segmentation files 
seg_files = os.listdir(SEGMENTATION_PATH)
# keep only files that end with .npy
seg_mask_files = [x for x in seg_files if x.endswith(".npy")]


############ GIF ############
# List all processed gif files with original video
gif_files = os.listdir(GIF_PATH)
# keep only .gif files that end with _processed
gif_files = [x for x in gif_files if x.endswith("_processed.gif")]


############ ALLIGEN FILES ############

# Load gif_files and seg_mask_files into separate pandas dataframes
df_gif_files = pd.DataFrame(gif_files, columns=["file_name_gif"])
df_seg_mask_files = pd.DataFrame(seg_mask_files, columns=["file_name_mask"])

# Split file_name to get patient id
df_gif_files["patient_id"] = df_gif_files["file_name_gif"].apply(lambda x: x.split("_")[0])
df_seg_mask_files["patient_id"] = df_seg_mask_files["file_name_mask"].apply(lambda x: x.split("_")[0])

# Split file_name to get video id
df_gif_files["video_id"] = df_gif_files["file_name_gif"].apply(lambda x: x.split("_")[1:3])
df_gif_files["video_id"] = df_gif_files["video_id"].apply(lambda x: "_".join(x))

df_seg_mask_files["video_id"] = df_seg_mask_files["file_name_mask"].apply(lambda x: x.split("_")[1:3])
df_seg_mask_files["video_id"] = df_seg_mask_files["video_id"].apply(lambda x: "_".join(x))

# Merge dataframes
df_gif_seg_mask = pd.merge(df_gif_files, df_seg_mask_files, on=["patient_id", "video_id"])
# Drop nan values
df_gif_seg_mask = df_gif_seg_mask.dropna()

# Check if the file number is within the range of the dataframe
if FILE_NO > len(df_gif_seg_mask):
    print("The file number is out of range. Please choose a number between 0 and {}.".format(len(df_gif_seg_mask)))
    exit()

gif_file_name = df_gif_seg_mask["file_name_gif"].iloc[FILE_NO]
mask_file_name = df_gif_seg_mask["file_name_mask"].iloc[FILE_NO]

patient_id = df_gif_seg_mask["patient_id"].iloc[FILE_NO]
video_id = df_gif_seg_mask["video_id"].iloc[FILE_NO]

save_file = os.path.join(SAVE_PATH, f"cubicial_diagrams_{patient_id}_{video_id}_{FILE_NO}.pickle")

# Check if the file has already been processed. If yes, exit the program. If not, continue loading the file.
if os.path.exists(save_file) and EXCLUDE_PROCESSED == True:
    print("The file has already been processed.")
    exit()

############ LOAD FILES ############

# Load the segmentation mask
print("Loading the segmentation mask number {}: {}".format(FILE_NO, mask_file_name))
mask_array = np.load(os.path.join(SEGMENTATION_PATH, mask_file_name))

# Load the gif file
print("Loading the gif file number {}: {}".format(FILE_NO, gif_file_name))
gif_list = load_gif(os.path.join(GIF_PATH, gif_file_name))

n_frames_gif = len(gif_list)
n_frames_mask = len(mask_array)

# Check if the frame number is not equal
if not n_frames_gif == n_frames_mask:
    print(f"Frame number is not equal. Gif has {n_frames_gif} frames and Mask has {n_frames_mask} frames")
    print("CROPPING TO MINIAL NUMBER OF FRAMES!")
    
    # Crop gif_list and mask_array to the same length
    min_frames = min(n_frames_gif, n_frames_mask)

    gif_list = gif_list[:min_frames]
    mask_array = mask_array[:min_frames]

    # Print shape of both arrays after cropping
    print(f"Gif shape: {gif_list.shape}")
    print(f"Mask shape: {mask_array.shape}")

############ FILTER & RESIZE ############

classes = [1, 2, 3, 4]
# Filter all frames
filtered_frames = []
for i in range(len(gif_list)):
    filtered_frames.append(filter_frame(gif_list[i], mask_array[i], resize_factor=SCALE_FACTOR))

# Stack all frames of the filtered & resized classes into a single array
stacked_filtered_frames = {}
for c in classes:
    stacked_filtered_frames[c] = np.stack([x[c] for x in filtered_frames])

# Crop all volumes
cropped_filtered_frames = {}
for c in classes:
    cropped_filtered_frames[c] = trim_zeros(stacked_filtered_frames[c])

############ CALCULATE PERSISTENCE DIAGRAMS ############

# Extract the persistence diagrams using giotto-tda
# Track connected components, loops, and voids
homology_dimensions = [0, 1, 2]

# Initialize the cubical persistence object
cubical_persistence = CubicalPersistence(
    homology_dimensions=homology_dimensions,
    n_jobs=-1,
)

# Calculate the persistence diagrams for all classes
print("Calculating the persistence diagrams")
# Take the time it takes to calculate the persistence diagrams
start = time.time()

class_cubical_persistence = {}

for c in classes:
    # We have to prepend a an axis to the array, because the fit_transform method expects the first diemnsion to be the number of samples
    # We have to do it indivudually for each class, because the shape of the arrays is different due to the cropping
    class_cubical_persistence[c] = cubical_persistence.fit_transform(cropped_filtered_frames[c][np.newaxis, :, :, :])

end = time.time()
# Print the time it took to calculate the persistence diagrams in minutes
print("Time to calculate the persistence diagrams: {} minutes".format((end - start) / 60))

# Save the persistence diagrams
print("Saving the persistence diagrams")
import pickle
with open(save_file, "wb") as f:
    pickle.dump(class_cubical_persistence, f)

print("Done")