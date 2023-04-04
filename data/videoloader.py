from .dataloader import DataLoader
import tensorflow as tf


class VideoLoader(DataLoader):

    def __init__(self, dataset_directory, index_df, label_df, num_frames, frame_rate, stride=5, cross_val_splits = 5, seed=42):
        super().__init__(dataset_directory, index_df, label_df, cross_val_splits, seed)
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.stride = stride

    def stride_videos(self, df):
        grp =  df.groupby(["Patient ID", "Bluepoint"])

        video_list = []
        label_list = []
        for _, group in grp:
            for i in range(0, group.shape[0], self.stride):
                end = i + (self.num_frames * self.frame_rate)
                if end > group.shape[0]:
                    break
                video_list.append(group.iloc[i:end:self.frame_rate])
                label_list.append(group["Label"].iloc[i])

        # Unpack df to just store the video frame paths in a list
        video_list = [vid_df["path_abs"].values for vid_df in video_list]
        return video_list, label_list

    def get_tf_dataset(self, test_fold_index, shuffle=True, seed=None):

        """
        Get a video tensorflow dataset for the data.
        :param test_fold_index: index of the test fold
        :return: ds_train, ds_test tensorflow datasets
        """
        # Create stride view of the data
        #strided_df = self.stride_frames(self.index_df, self.stride, start=0, end=None)
        # Get the train and test folds
        train_df, test_df = self.get_folds(self.index_df, test_fold_index)

        file_paths_train, labels_train = self.stride_videos(train_df)
        file_paths_test, labels_test = self.stride_videos(test_df)

        print("TODO: Implement shuffling")

        # Create tensorflow dataset
        ds_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
        ds_test = tf.data.Dataset.from_tensor_slices((file_paths_test, labels_test))

        # Map the file paths to tensors
        ds_train = ds_train.map(lambda x, y: tf.py_function(self.read_video_tf, [x, y], [tf.float32, tf.int32]))
        ds_test = ds_test.map(lambda x, y: tf.py_function(self.read_video_tf, [x, y], [tf.float32, tf.int32]))

        return ds_train, ds_test

    def map_file_path_to_tf_object(self, file_path, label):
        return self.read_video_tf(file_path, label)


    def read_video_tf(self, frame_files, label):
        """
        Reads an image and returns it as a tensor.
        :param image_file: path to the image file
        :param label: label of the image
        :return: tensor containing the image
        """

        tensor_list = []
        for file in frame_files:
            frame = tf.io.read_file(file)
            frame = tf.image.decode_jpeg(frame, channels=3)
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            #image_file = tf.image.resize(image_file, [256, 256])
            tensor_list.append(frame)
        video_tensor = tf.stack(tensor_list, axis=0)
        return video_tensor, label

