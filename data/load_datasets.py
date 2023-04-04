import os
from xml.etree.ElementInclude import include
import pandas as pd
from data.imageloader import ImageLoader
from pathlib import Path



def get_maastricht_loader(label_column="clin_diagn#COVID19_pneumonia"):
    print("Creating maastricht loader")
    if os.environ["USER"] == "matthiasrichter":
        #Local Matthias
        DRIVE_LOCATION = "/Volumes"
    elif os.environ["USER"] == "mrichte":
        #Remote Matthias
        DRIVE_LOCATION = "/itet-stor"
    else:
        raise ValueError("Unknown user")
    INDEX_LOCATION = os.path.join(DRIVE_LOCATION, "mrichte/covlus_bmicnas02/maastricht_image_dataset/index.csv")
    LABEL_LOCATION = os.path.join(DRIVE_LOCATION, "mrichte/covlus_bmicnas02/clinical_data.csv")
    DATASET_LOCATION = os.path.join(DRIVE_LOCATION, "mrichte/covlus_bmicnas02/maastricht_image_dataset/")


    # Load index and label data
    idx_df = pd.read_csv(INDEX_LOCATION)
    label_df = pd.read_csv(LABEL_LOCATION)
    assert label_column in label_df.columns, f"Invalid label column. Valid columns are {label_df.columns}"

    # Rename patient id to be the same
    idx_df.rename(columns={"patient": "Patient ID"}, inplace=True)
    idx_df.rename(columns={"frame": "Frame"}, inplace=True)
    idx_df.rename(columns={"bluepoint": "Bluepoint"}, inplace=True)
    label_df.rename(columns={"Video ID": "Patient ID"}, inplace=True)
    label_df.rename(columns={label_column: "Label"}, inplace=True)

    label_df = label_df[["Patient ID", "Label"]]
    idx_df = idx_df.join(label_df.set_index("Patient ID"), on="Patient ID")
    idx_df.drop(columns=["path_abs"], inplace=True)

    IL = ImageLoader(DATASET_LOCATION, idx_df, cross_val_splits = 5, seed=42)

    return IL

def get_pocovid_loader(include_labels=["Cov", "Reg"], data_dir=None):
    possible_labels = ["Cov", "Reg", "Vir", "Pne"]
    assert set(include_labels).issubset(set(possible_labels)), f"Invalid label selection. Valid labels are {possible_labels}"

    print("Creating pocovid loader")
    if os.environ["USER"] == "matthiasrichter":
        #Local Matthias
        DATA_DIR = "/Users/matthiasrichter/Library/CloudStorage/OneDrive-Personal/Studium/Masterthesis/covid19_ultrasound/data/"
    elif os.environ["USER"] == "mrichte":
        #Remote Matthias
        DATA_DIR = "/itet-stor/mrichte/covlus_bmicnas02/additional_datasets/POCOVID/data/"
    else:
        assert data_dir is not None, "No data dir provided and User not know. Cannot load data."
        DATA_DIR = data_dir
    
    FRAME_DIR = os.path.join(DATA_DIR, 'frame_dataset')


    path_list = []
    for path in Path(FRAME_DIR).rglob('*.jpg'):
        stem = path.stem
        stem_split = stem.split("_", 2)
        frame = int(stem_split[0])
        label = stem_split[1]
        video_name = stem_split[2]
        path_abs = path.as_posix()
        path_relative = path.relative_to(FRAME_DIR).as_posix()

        path_list.append({"dataset": "pocovid",
                        "video_name": video_name,
                        "Frame": frame,
                        "Label": label,
                        "path_relative": path_relative,
                        "path_abs": path_abs,})

    idx_df = pd.DataFrame(path_list)

    patient_df = idx_df.groupby('video_name').first().reset_index().reset_index()
    patient_df.rename(columns={'index': 'Patient ID'}, inplace=True)
    patient_df = patient_df[["Patient ID", "video_name"]]

    df = idx_df.join(patient_df.set_index('video_name'), on='video_name').sort_values(by=['Patient ID', 'Frame'], ascending=False).reset_index(drop=True)
    # Renaming the columns to match the uniform format
    df.loc[df['Label'] == 'pne','Label'] = 'Pne'
    # Add dummy Bluepoint var
    df["Bluepoint"] = "None"

    # Select only covid and normal for now
    FILTER_DIAGNOSIS = include_labels 
    df = df[df.Label.isin(FILTER_DIAGNOSIS)].reset_index(drop=True)

    # Need to exclude some columns because tf.image loader cant handle the names
    FILTER_NAMES = ["Cov-Atlas+(45)", "Cov-Atlas+(44)", "Cov-Atlas-+(43)", "Cov-Atlas-Day+1"]
    df = df[~df.video_name.isin(FILTER_NAMES)].reset_index(drop=True)

    IL = ImageLoader(FRAME_DIR, df, cross_val_splits = 5)

    return IL


def get_covid_us_loader(include_labels=["Cov", "Reg"], data_dir=None):
    possible_labels = ["Cov", "Reg", "other", "Pne"]
    assert set(include_labels).issubset(set(possible_labels)), f"Invalid label selection. Valid labels are {possible_labels}"

    print("Creating covid_us loader")
    if os.environ["USER"] == "matthiasrichter":
        #Local Matthias
        DATA_DIR = "/Users/matthiasrichter/Studium_local/MA_local/COVID-US/data/image/clean"
    elif os.environ["USER"] == "mrichte":
        #Remote Matthias
        DATA_DIR = "/itet-stor/mrichte/covlus_bmicnas02/additional_datasets/COVIDxUS/data/image/clean"
    else:
        assert data_dir is not None, "No data dir provided and User not know. Cannot load data."
        DATA_DIR = data_dir
    
    # List files
    file_list = os.listdir(DATA_DIR)
    df_file_names = pd.DataFrame(file_list, columns=["path_relative"])

    df_info = df_file_names["path_relative"].str.split("_", expand=True)
    df_info.columns = ["Patient ID", "Source", "Label", "Test", "Probe type", "Cropped", "Frame"]
    df_info["Frame"] = df_info["Frame"].str.split(".", expand=True)[0].str.extract('(\d+)').astype(int)
    df_info["Patient ID"] = df_info["Patient ID"].astype(int)
    df = pd.concat([df_info, df_file_names], axis=1)
    df = df.sort_values(["Patient ID", "Frame"]).reset_index(drop=True)
    df["Bluepoint"] = "None"

    # Make label names consistent with pocovid dataset
    df.loc[df['Label'] == 'normal','Label'] = 'Reg'
    df.loc[df['Label'] == 'pneumonia','Label'] = 'Pne'
    df.loc[df['Label'] == 'covid','Label'] = 'Cov'

    # Select only covid and normal for now
    FILTER_DIAGNOSIS = include_labels
    df = df[df.Label.isin(FILTER_DIAGNOSIS)].reset_index(drop=True)

    FILTER_PROBE = ["convex"]
    df = df[df['Probe type'].isin(FILTER_PROBE)].reset_index(drop=True)

    IL = ImageLoader(DATA_DIR, df, cross_val_splits = 5)

    return IL

def get_pocovid_covid_us_combined_dataset():
    pocovid_loader = get_pocovid_loader()
    covid_us_loader = get_covid_us_loader()
    combined_loader = ImageLoader.combine_dataloaders(pocovid_loader, covid_us_loader, "pocovid", "covid_us")
    return combined_loader

if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    print("Tensorflow version: ", tf.__version__)
    print("TF versions must be >= 2.4")
    print("Testing pocovid loader")
    pocovid_loader = get_pocovid_loader()
    print("Pocovid loader index df:")
    print(pocovid_loader.index_df.head())
    print("Striding pocovid loader")
    pocovid_loader.stride(5)
    print("Making tf dataset")
    train, test = pocovid_loader.get_tf_dataset(0)
    testX, testY = tf.data.Dataset.get_single_element(test.batch(len(test)))
    testX = testX.numpy()
    testY = testY.numpy()
    num_classes = len(np.unique(testY))


DATASET_FACTORY = {
    'maastricht': get_maastricht_loader,
    'pocovid': get_pocovid_loader,
    'covid_us': get_covid_us_loader,
    'combined': get_pocovid_covid_us_combined_dataset,
    'pocovid_covid_us_combined': get_pocovid_covid_us_combined_dataset,
}

def get_dataset_loader(dataset_name):
    assert dataset_name in DATASET_FACTORY, f"Invalid dataset name. Valid names are {DATASET_FACTORY.keys()}"
    return DATASET_FACTORY[dataset_name]()