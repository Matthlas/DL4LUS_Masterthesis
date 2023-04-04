print("Importing libraries...")
import os
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd


def get_mask(file):
    frame_list = []
    with Image.open(os.path.join(segmentation_path, file)) as im:
        im.seek(1)  # skip to the second frame

        try:
            while 1:
                im.seek(im.tell() + 1)
                frame = im.copy()
                # # Convert to RGB
                # frame = frame.convert("RGB")
                frame_list.append(np.array(frame))
        except EOFError:
            pass  # end of sequence
    
    return np.array(frame_list)


def mask_2_segmentation(mask):

    # Convert to numpy array
    frames = np.array(mask)

    # Create empty list for all segmentation maps and pixel counts
    seg_cluster_maps = []
    pixel_counts = []

    # Iterate over all frames
    for i, frame in enumerate(frames):
        print(f"Processing frame {i+1}/{frames.shape[0]}")

        # Creare empty list for segmentation and pixel count
        seg_cluster_map = []
        pixel_count = {"blue": 0, "green": 0, "orange":0, "red": 0, "black": 0}

        # Reshape to list of colors
        colors = frame.reshape((frame.shape[0] * frame.shape[1],4))

        # Iterate over all pixels
        for color in colors:

            # if blue
            if color[0] == 0 and color[1] == 0 and color[2] > 0:
                seg_cluster_map.append(1)
                pixel_count["blue"] += 1
            # if green
            elif color[0] == 0 and color[1] > 0 and color[2] == 0:
                seg_cluster_map.append(2)
                pixel_count["green"] += 1
            # if orange
            elif color[0] > 0 and color[1] > 0 and color[2] == 0:
                seg_cluster_map.append(3)
                pixel_count["orange"] += 1
            # if red
            elif color[0] > 0 and color[1] == 0 and color[2] == 0:
                seg_cluster_map.append(4)
                pixel_count["red"] += 1
            # else black
            else:
                seg_cluster_map.append(0)
                pixel_count["black"] += 1

        # Reshape to original image shape
        seg_cluster_map = np.array(seg_cluster_map).reshape((frame.shape[0], frame.shape[1]))

        # Append segmentation map and pixel count to list
        seg_cluster_maps.append(seg_cluster_map)
        pixel_counts.append(pixel_count)


    return pixel_counts, seg_cluster_maps

def save_np_as_gif(array, file_name):
    # Convert np array to PIL images
    imgs = [Image.fromarray(img) for img in array]
    # Save as gif
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(file_name, save_all=True, append_images=imgs[1:], duration=50, loop=0)




if __name__ == "__main__":
    # Path to segmentation masks
    segmentation_path = "/itet-stor/mrichte/covlus_bmicnas02/cropped_videos_segmented"

    # List all files & directories
    seg_files = os.listdir(segmentation_path)

    # keep only .gif files that end with _mask
    seg_mask_files = [x for x in seg_files if x.endswith("mask.gif")]

    print(f"Found {len(seg_mask_files)} segmentation masks")
    
    collected_pixel_counts = []
    for file in seg_mask_files:
        print(f"Processing file {file}")
        mask = get_mask(file)
        pixel_counts, seg_cluster_maps = mask_2_segmentation(mask)
        print(pixel_counts)
        print(seg_cluster_maps)

        save_path = os.path.join(segmentation_path, "post_processing")
        # Save segmentation maps as gif
        save_np_as_gif(seg_cluster_maps, os.path.join(save_path, file.replace("_mask.gif", "_seg.gif")))

        # Save pixel counts as npy
        np.save(os.path.join(save_path, file.replace("_mask.gif", "_pixel_counts.npy")), pixel_counts)

        # Append pixel counts to list
        collected_pixel_counts.append(pixel_counts)

    # Save all pixel counts as pandas dataframe
    df = pd.DataFrame(collected_pixel_counts)
    df.to_csv(os.path.join(save_path, "pixel_counts.csv"))