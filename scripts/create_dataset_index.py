import os
import pandas as pd
from pathlib import Path
import sys
import argparse
from loguru import logger
import logging



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger.info('Creating image dataset index')

ap = argparse.ArgumentParser()
ap.add_argument(
    '-d', '--directory', help='dataset directory', default="/itet-stor/mrichte/covlus_bmicnas02/maastricht_image_dataset/"
)
args = vars(ap.parse_args())

IMG_DIR = args['directory']

path_list = []
for path in Path(IMG_DIR).rglob('*.jpg'):
    stem = path.stem
    stem_split = stem.split("_", 1)
    frame = int(stem_split[0])
    video_name = stem_split[1]
    bluepoint = path.parent.parent.name
    patient = path.parent.parent.parent.name
    dataset = path.parent.parent.parent.parent.name
    path_abs = path.as_posix()
    path_relative = path.relative_to(IMG_DIR).as_posix()

    path_list.append({"dataset": dataset, 
                    "patient": patient, 
                    "bluepoint": bluepoint, 
                    "video_name": video_name,
                    "frame": frame,
                    "path_relative": path_relative,
                    "path_abs": path_abs,})

idx_df = pd.DataFrame(path_list)
idx_df.to_csv(os.path.join(IMG_DIR, "index.csv"), index=False)
logger.info(f'Done. Created index.csv in {IMG_DIR}')