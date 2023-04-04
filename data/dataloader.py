import os
import pandas as pd
import tensorflow as tf
import numpy as np
from collections import defaultdict
from collections import ChainMap
from abc import ABC, abstractmethod

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

class DataLoader(ABC):
    def __init__(self, dataset_directory, index_df, cross_val_splits = 5, seed=42, one_hot=True):
        """
        Initialize the DataLoader class.
        :param index_df: pandas dataframe containing a list of all the image paths. Needs "Patient ID", "Bluepoint", "path_relative", "Label", "Frame" column.
        :param label_df: pandas dataframe containing labels for the data. Needs "Patient ID" column.
        :param cross_val_splits: number of cross validation splits
        :param batch_size: batch size for the dataloader
        :param shuffle: whether to shuffle the data
        :param seed: seed for the dataloader
        """

        assert "Patient ID" in index_df.columns, "index_df needs a 'Patient ID' column."
        assert "Label" in index_df.columns, "index_df needs a 'Label' column."
        assert "Bluepoint" in index_df.columns, "index_df needs a 'Bluepoint' column."
        assert "path_relative" in index_df.columns, "index_df needs a 'path_relative' column."
        assert "Frame" in index_df.columns, "index_df needs a 'Frame' column."

        # Set class variables
        self.dataset_directory = dataset_directory
        self.index_df = index_df
        # self.label_df = label_df
        self.cross_val_splits = cross_val_splits
        self.seed = seed
        self.one_hot = one_hot
        self.index_df.sort_values(by=["Patient ID", "Bluepoint", "Frame"], inplace=True)

        self.class_names = None
        # One hot encode labels
        if self.one_hot:
            self.onehottyfy_labels()

        if "path_abs" in self.index_df.columns:
            print("path_abs column already exists in index_df. Using old column. If you want to update the paths, drop the column and rerun the script.")
        else:
            # Change absolute path relative to the system
            self.index_df["path_abs"] = self.index_df.apply(lambda row: os.path.join(self.dataset_directory, row.path_relative), axis=1)

        # Create patient level dataframe
        self.patient_df = self.index_df.groupby("Patient ID").first().reset_index()

        if "Fold" in self.patient_df.columns:
            print("Dataframe already has a 'Fold' column. Using that as cross valitation.")
        else:
            # Create cross validation folds
            self.create_cross_val_folds(cross_val_splits)

        self.num_of_frames = self.index_df.groupby(["Patient ID", "Bluepoint"]).apply(lambda x: len(x))


    def __len__(self):
        return(len(self.index_df))

    @abstractmethod
    def get_tf_dataset(self, test_fold_index, shuffle=True, seed=None, return_dataframe=False):
        """
        Get a tensorflow dataset (generator) for the data.
        :param test_fold_index: index of the test fold
        :return: ds_train, ds_test tensorflow datasets
        """
        pass

    @abstractmethod
    def map_file_path_to_tf_object(self, file_path, label):
        """
        Map a file path to a tensorflow object.
        :param file_path: file path of the image or list of file paths to the frames
        :param label: label of the image or video
        :return: tensorflow object
        """
        pass

    def onehottyfy_labels(self):
        """
        One hot encode the labels.
        """
        print("One hot encoding labels...")
        # Binarizing labels
        lb = LabelBinarizer()
        lb.fit(self.index_df["Label"])
        labels = lb.transform(self.index_df["Label"])

        # If categorical, convert to one hot categorical
        num_classes = len(set(self.index_df["Label"]))
        if num_classes == 2:
            labels = to_categorical(labels, num_classes=num_classes)

        # Add one hot encoded labels to label_df, create new column for label mappings
        self.index_df["Label_one_hot"] = pd.DataFrame({"Label" : list(labels)})["Label"]
        self.class_names = lb.classes_
        
        # print(f'Total of {num_classes} classes. Class mappings are:', lb.classes_)

    def get_folds(self, df, test_fold_index=0):
        """
        Get the test fold with index test_fold_index and the rest as train folds.
        :param df: dataframe with "Fold" column
        :param test_fold_index: index of the test fold
        :return: train folds and test fold
        """
        train = df[df["Fold"] != test_fold_index]
        test = df[df["Fold"] == test_fold_index]
        return train, test

    def create_cross_val_folds(self, cross_val_splits):
        """
        Create cross validation folds.
        :param cross_val_splits: number of cross validation splits
        :return: list of lists with the fold indices
        """
        print("Creating cross validation folds...")
        # Count the number of patients in each class
        label_counts = self.patient_df["Label"].value_counts()
        label_counts = pd.DataFrame(label_counts).reset_index().rename(columns={"index": "Label", "Label": "Count"})
        # For each class get a list of Patient IDs belonging to that class
        label_counts["participants"] = label_counts.Label.apply(lambda x: self.patient_df[self.patient_df["Label"] == x]["Patient ID"].values)
        # Create permutations of those
        np.random.seed(42)
        label_counts["permuted"] = label_counts.participants.apply(lambda x: np.random.permutation(x))
        # Create a list [0,1,2,3,4,0,1,2,3,4,0,1...] in the length of the number of patients in each class as a mapping to the cross validation splits
        l = list(range(cross_val_splits))*int(label_counts.Count.max()/cross_val_splits+1)
        # Shorten that list to the length of the number of patients in each respective class and add to df
        label_counts["fold_idx"] = label_counts.Count.apply(lambda x: l[:x])
        # Create a dictionary mapping the Patient ID to the fold index
        label_counts["id_to_fold"] = label_counts.apply(lambda row: dict(zip(row.permuted, row.fold_idx)), axis=1)
        # Combine all dictionaries into one
        id_to_fold = dict(ChainMap(*label_counts.id_to_fold.values))
        # Create a column in the patient_df that maps each patient to a fold
        self.patient_df["Fold"] = self.patient_df["Patient ID"].map(id_to_fold)
        # Create a column in the index_df that maps each patient to a fold
        self.index_df["Fold"] = self.index_df["Patient ID"].map(id_to_fold)
        print(f"Created cross validation using {len(label_counts)} classes with:", *zip(label_counts.Label.values, label_counts.Count.values), "cases per class.")

    def combine_dataloaders(DL1, DL2, name1, name2):
        """
        Combine two dataloaders into one.
        :param DL1: dataloader 1
        :param DL2: dataloader 2
        :return: combined dataloader
        """
        assert type(DL1) == type(DL2), "Dataloaders must be of the same type."
        DL1_df = DL1.index_df.copy()
        DL2_df = DL2.index_df.copy()
        # DL1_df.drop(columns=["Label_one_hot"], inplace=True)
        # DL2_df.drop(columns=["Label_one_hot"], inplace=True)
        DL1_df["dataset"] = name1
        DL2_df["dataset"] = name2

        assert set(DL1_df.Label.unique()) == set(DL2_df.Label.unique()), "Labels are not the same in the two dataloaders. Make sure Label columns contain the same labels."

        df_combined = pd.concat([DL2_df, DL1_df], axis=0)
        df_combined["Patient ID"] = df_combined["Patient ID"].astype(str) + "_" + df_combined["dataset"]

        return type(DL1)(None, df_combined, seed=DL1.seed, one_hot=DL1.one_hot)

