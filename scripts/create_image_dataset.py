import os
import pandas as pd
import cv2
import logging
import sys
import argparse
from loguru import logger



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger.info('Creating image dataset')

ap = argparse.ArgumentParser()
ap.add_argument(
    '-vd', '--video_dir', help='initial video directoy', default="/itet-stor/mrichte/covlus_bmicnas02/cropped_videos/version1"
)
ap.add_argument(
    '-ld', '--label_directory', help='fold to take as test data', default="/itet-stor/mrichte/covlus_bmicnas02/clinical_data.csv"
)
ap.add_argument(
    '-md', '--mapping_directory', help='fold to take as test data', default="/itet-stor/mrichte/covlus_bmicnas02/cropped_videos/video_patient_mapping.csv"
)
ap.add_argument(
    '-bd', '--bluepoint_directory', help='fold to take as test data', default="/itet-stor/mrichte/covlus_bmicnas02/raw/bluepoints.csv"
)
ap.add_argument(
    '-od', '--out_directory', help='fold to take as test data', default="/itet-stor/mrichte/covlus_bmicnas02/maastricht_image_dataset/"
)
args = vars(ap.parse_args())

VID_DIR = args['video_dir']
LABEL_DIR = args['label_directory']
MAPPING_CSV = args['mapping_directory']
BLUEPOINT_CSV = args['bluepoint_directory']
OUT_DIR = args['out_directory']


# List all video files, get their associated Patient ID from filename
video_files = os.listdir(VID_DIR)
clinical_data_csv = pd.read_csv(LABEL_DIR) 

video_file_path_df = pd.DataFrame(video_files, columns = ['Video File'])
id_df = pd.DataFrame([path.split("_") for path in video_files], columns = ["Patient ID", "image","Video Name","none"])
id_df.drop(columns = ["image","none"], inplace = True)
video_file_path_df = pd.concat([video_file_path_df, id_df], axis = 1)

# Merge with clinical data
data = pd.merge(clinical_data_csv, video_file_path_df, left_on="Video ID", right_on="Patient ID")

# Select only columns needed for mapping
mapping_df = data[["Patient ID", "Video File"]].copy()
# Add entire path
mapping_df["Video Path"] = mapping_df["Video File"].apply(lambda x: os.path.join(VID_DIR, x))
# Add video name column for file naming
mapping_df["Video name"] =  mapping_df["Video File"].str.split(".", expand=True)[0]
mapping_df["Video name"] = mapping_df["Video name"].str.split("_", 1, expand=True,)[1]

# Add bluepoint information
bluepoints = pd.read_csv(BLUEPOINT_CSV)
bluepoints = pd.concat((bluepoints, bluepoints["Patient"].str.split("_", expand=True)[[1]]), axis=1)
bluepoints = bluepoints.rename(columns = {1: "Patient ID"})
bluepoints["Video name"] = bluepoints["Video file"].str.split("." , expand=True)[0]
bluepoints = bluepoints.drop(columns = ["Patient", "Video file", "Patient ID"])

mapping_df = pd.merge(mapping_df, bluepoints, on = "Video name")
# Some Bluepoints are nan, fill with str representation "None"
mapping_df["Blue point"].fillna("None", inplace = True)


for idx, row in mapping_df.iterrows():
    video_path = row["Video Path"]
    patient = row["Patient ID"]
    bluepoint = row["Blue point"]
    video_name = row["Video name"]

    if not os.path.exists(os.path.join(OUT_DIR, patient)):
        os.makedirs(os.path.join(OUT_DIR, patient))
    if not os.path.exists(os.path.join(OUT_DIR, patient, bluepoint)):
        os.makedirs(os.path.join(OUT_DIR, patient, bluepoint))
    if not os.path.exists(os.path.join(OUT_DIR, patient, bluepoint, "images")):
        os.makedirs(os.path.join(OUT_DIR, patient, bluepoint, "images"))

    logger.info(f"Processing Patient {patient}, Bluepoint {bluepoint} (Row {idx} of {len(mapping_df)})")
    # Opens the Video file and writes all the frames to the images folder
    cap= cv2.VideoCapture(video_path)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(os.path.join(OUT_DIR, patient, bluepoint, "images", str(i).zfill(3) + "_" + video_name + ".jpg"), frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()

logger.info(f"Finished sucessfully")