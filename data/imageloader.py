from .dataloader import DataLoader
import tensorflow as tf
import numpy as np

class ImageLoader(DataLoader):

    def __init__(self, dataset_directory, index_df, cross_val_splits = 5, seed=42, one_hot=True):
        super().__init__(dataset_directory, index_df, cross_val_splits, seed, one_hot)
        self.complete_index_df = self.index_df.copy()


    def _stride_frames(self, df, stride, start=0, end=None):
        """
        Create a strided view of all videos in the dataframe.
        :param stride: stride of the view
        :param start: start index of the view
        :param end: end index of the view
        :return: strided view of the dataframe
        """
        return df.groupby(["Patient ID", "Bluepoint"]).apply(lambda x: x.iloc[start:end:stride]).reset_index(drop=True)

    def stride(self, stride):
        self.complete_index_df.sort_values(by=["Patient ID", "Bluepoint", "Frame"], inplace=True)
        print("Striding frames with stride: ", stride)
        self.index_df = self._stride_frames(self.complete_index_df, stride)

    def random_sample(self, random_frac, seed=None):
            self.index_df = self.complete_index_df.groupby(["Fold"]).sample(frac=random_frac, random_state=seed).reset_index(drop=True)


    def get_tf_dataset(self, test_fold_index, shuffle=True, seed=None, return_dataframe=False):
        train_df, test_df = self.get_folds(self.index_df, test_fold_index)

        if shuffle:
            train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        file_paths_train = train_df["path_abs"].values
        file_paths_test = test_df["path_abs"].values
        if self.one_hot:
            labels_train = train_df["Label_one_hot"].values
            labels_test = test_df["Label_one_hot"].values
            labels_train = [np.asarray(x).astype('float32') for x in labels_train]
            labels_test = [np.asarray(x).astype('float32') for x in labels_test]
        else:
            labels_train = train_df["Label"].values
            labels_test = test_df["Label"].values

        # Create tensorflow dataset
        ds_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
        ds_test = tf.data.Dataset.from_tensor_slices((file_paths_test, labels_test))

        # Map the file paths to tensors
        ds_train = ds_train.map(self.map_file_path_to_tf_object, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(self.map_file_path_to_tf_object, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if return_dataframe:
            return ds_train, ds_test, train_df, test_df
        else:
            return ds_train, ds_test

    def map_file_path_to_tf_object(self, file_path, label):
        return self.read_img_tf(file_path, label)

    def read_img_tf(self, image_file, label, normalize = False, resize=True, img_size = (224, 224)):
        """
        Reads an image and returns it as a tensor.
        :param image_file: path to the image file
        :param label: label of the image
        :return: tensor containing the image
        """
        image_file = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image_file, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if resize:
            image = tf.image.resize(image, img_size)
        if normalize:
            image = image / 255.0
        return image, label


    def get_test_data(self, test_fold_index):
        _, test_df = self.get_folds(self.index_df, test_fold_index)

        file_paths_test = test_df["path_abs"].values
        if self.one_hot:
            labels_test = test_df["Label_one_hot"].values
            labels_test = [np.asarray(x).astype('float32') for x in labels_test]
        else:
            labels_test = test_df["Label"].values

        # Create tensorflow dataset
        test_ds = tf.data.Dataset.from_tensor_slices((file_paths_test, labels_test))

        # Map the file paths to tensors
        test_ds = test_ds.map(self.map_file_path_to_tf_object, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Load the data
        testX, testY = tf.data.Dataset.get_single_element(test_ds.batch(len(test_ds)))
        testX = testX.numpy()
        testY = testY.numpy()
        
        return testX, testY, test_df