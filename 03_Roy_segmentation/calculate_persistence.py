import argparse
import os
import numpy as np
import time
from skimage import measure # For marching cubes
from gtda.homology import VietorisRipsPersistence # For persistence diagrams
from tda_utils import scale_down, get_classes, normalize_pc # Own functions

# Parse the arguments
parser = argparse.ArgumentParser()
# Path to the post-processed segmentation masks (npy files, shape: (n_frames, X, Y), should only contain int values 0-4 representing the different segmentations)
parser.add_argument("--segmentation_path", type=str, default="/itet-stor/mrichte/covlus_bmicnas02/cropped_videos_segmented/post_processing", help="Path to the segmentation masks")
# Number of the file to process
parser.add_argument("--file_no", type=int, help="Number of the file to process")
# Scale factor for the images
parser.add_argument("--scale_factor", type=float, default=0.05, help="Scale factor for the images")
# Save path
parser.add_argument("--save_path", type=str, default="/itet-stor/mrichte/covlus_bmicnas02/cropped_videos_segmented/persistence_diagrams", help="Path to save the results")
# Exclude already processed files
parser.add_argument("--exclude_processed", type=bool, default=True, help="Exclude already processed files")
args = parser.parse_args()

# Path to the segmentation masks
SEGMENTATION_PATH = args.segmentation_path
# Number of the file to process
FILE_NO = args.file_no
# Scale factor for the images
SCALE_FACTOR = args.scale_factor
# Save path
SAVE_PATH = args.save_path
# Exclude already processed files
EXCLUDE_PROCESSED = args.exclude_processed

# List all files & directories
seg_files = os.listdir(SEGMENTATION_PATH)
# keep only files that end with .npy
seg_mask_files = [x for x in seg_files if x.endswith(".npy")]

# Check if the file exists. If not, exit the program. If yes, define the file name and continue loading the file.
if FILE_NO >= len(seg_mask_files):
    print("The file does not exist.")
    exit()

file_name = seg_mask_files[FILE_NO]
save_file = os.path.join(SAVE_PATH, f"diagrams_{file_name}_{FILE_NO}.npy")

# Check if the file has already been processed. If yes, exit the program. If not, continue loading the file.
if os.path.exists(save_file) and EXCLUDE_PROCESSED == True:
    print("The file has already been processed.")
    exit()

# Load the segmentation mask
print("Loading the segmentation mask number {}: {}".format(FILE_NO, file_name))
array = np.load(os.path.join(SEGMENTATION_PATH, file_name))

# Calculate the size of the scaled images based on the scale factor
size = (int(array.shape[1] * SCALE_FACTOR), int(array.shape[2] * SCALE_FACTOR))
print("Rescaling to size: {}".format(size))

# Scale down all images in the array
# Create a new array with the scaled images
scaled_array = np.empty((array.shape[0], size[0], size[1]))

for i in range(array.shape[0]):
    scaled_array[i] = scale_down(array[i], size)

# Extract the segmentations as individual arrays stroed in a dictionary with the cluster number as key
# {1: cluster1 (A-Lines) np.array (n_frames, *size), 2: ...}
classes = get_classes(scaled_array)

# List of all segmentations numbers
class_numbers = [1, 2, 3, 4]

# Extract the isosurface of all segmentations using marching cubes
print("Extracting the isosurfaces of all segmentations")
vertices = {}
faces = {}
normals = {}
values = {}
for i in class_numbers:
    # Extract the isosurface
    try:
        verts, face, norm, val = measure.marching_cubes(classes[i])
        # Store the isosurface in the dictionary
        vertices[i] = verts
        faces[i] = face
        normals[i] = norm
        values[i] = val
    except:
        print("No isosurface could be extracted for class {}".format(i))
        # Add point at origin to dictionary as placeholder
        vertices[i] = np.array([[0.0, 0.0, 0.0]])
        faces[i] = np.array([[0, 0, 0]])
        normals[i] = np.array([[0.0, 0.0, 0.0]])
        values[i] = np.array([0.0])


# Verticies are the surface points of the isosurface
# Normalize all vertices point clouds to have a mean of 0 and max distance of 1
print("Normalizing the surface point clouds")
vertices_normalized = [normalize_pc(value) for key, value in vertices.items()]

# Print the number of points in each point cloud
for key, value in vertices.items():
    print("\tNumber of points in class {}: {}".format(key, value.shape[0]))


# Extract the persistence diagrams using giotto-tda
# Track connected components, loops, and voids
homology_dimensions = [0, 1, 2]

# Initialize the vietoris rips persistence object
# Collapse edges to speed up H2 persistence calculation!
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=4,
    # n_jobs=-1,
    collapse_edges=True,
    max_edge_length=1.0,
)

# Calculate the persistence diagrams
print("Calculating the persistence diagrams")
# Take the time it takes to calculate the persistence diagrams
start = time.time()

diagrams = persistence.fit_transform(vertices_normalized)

end = time.time()
# Print the time it took to calculate the persistence diagrams in minutes
print("Time to calculate the persistence diagrams: {} minutes".format((end - start) / 60))

# Save the persistence diagrams
print("Saving the persistence diagrams")
np.save(save_file, diagrams)

print("Done")