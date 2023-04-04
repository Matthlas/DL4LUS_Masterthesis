import os
import pandas as pd

SAMPLE_PATH = "/Users/matthiasrichter/Library/CloudStorage/OneDrive-Personal/Studium/Masterthesis/Sample"

def get_drive_location():
    if os.environ["USER"] == "matthiasrichter":
        #Local Matthias
        drive_location = "/Volumes"
    elif os.environ["USER"] == "mrichte":
        #Remote Matthias
        drive_location = "/itet-stor"
    else:
        raise ValueError("Unknown user")
    return drive_location

def get_data_location():
    drive_location = get_drive_location()
    return os.path.join(drive_location, "mrichte/covlus_bmicnas02/")

def get_clinical_df():
    DATA_PATH = get_data_location()
    clinical_data_path = os.path.join(DATA_PATH,"clinical_data.csv")
    clinical_data = pd.read_csv(clinical_data_path)
    return clinical_data

def get_bluepoints_df():
    DATA_PATH = get_data_location()
    bluepoints_path = os.path.join(DATA_PATH,"raw/bluepoints.csv")
    bp = pd.read_csv(bluepoints_path)
    # Extract Patient ID & video_name
    bp["Patient ID"] = bp["Patient"].str.split("_", expand=True)[1]
    bp["video_name"] = bp["Video file"].str.replace(".mp4", "")
    bp.rename(columns={"Blue point": "Bluepoint"}, inplace=True)
    bp = bp.drop(columns=["Video file", "Patient"])
    return bp


def get_manual_severity_scores():
    DATA_PATH = get_data_location()
    manual_severity_score_path = os.path.join(DATA_PATH,"severity_scores.xlsx")

    # Read the excel sheets of both tablets
    tablets_dict_df = pd.read_excel(manual_severity_score_path, sheet_name=["Tablet A", "Tablet C"] )
    tablet_A = tablets_dict_df.get("Tablet A")
    tablet_C = tablets_dict_df.get("Tablet C")

    tablet_A["Tablet"] = "A"
    tablet_C["Tablet"] = "C"

    # Drop all unnamed columns
    tablet_A = tablet_A.loc[:, ~tablet_A.columns.str.contains('^Unnamed')]
    tablet_C = tablet_C.loc[:, ~tablet_C.columns.str.contains('^Unnamed')]

    # Change columns of tablet C to make them uniform
    tablet_C.rename(columns={"comment": "comments"}, inplace=True)

    # Combine the two dataframes
    severity_manual = pd.concat([tablet_A, tablet_C], axis=0)
    # Keep only columns with KEEP == YES
    severity_manual = severity_manual[severity_manual["KEEP"] == "YES"]
    # Print number of discarded rows
    print("Number of discarded rows:", len(tablet_A) + len(tablet_C) - len(severity_manual))
    # Drop file extension from video ID
    severity_manual["video_name"] = severity_manual["Video ID"].str.replace(".mp4", "")
    # Drop video ID, Patient ID, KEEP
    severity_manual = severity_manual.drop(columns=["Video ID", "Patient ID", "KEEP"])
    
    return severity_manual



