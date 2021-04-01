"""
data loading and transforming pandas dataframe to Tensorflow Dataset
"""

import pandas
import tensorflow as tf


class TensorflowDataMapper:
    """Transform pandas data frame to tensorflow data set
    """
    def __init__(self):
        pass

    def data_frame_to_dataset(self, data_frame_features=None, data_frame_labels=None):
        """Transform from data frame to data_set

        Args:
            data_frame_features (pandas.DataFrame): Features Data.
            data_frame_labels (pandas.DataFrame): Target Data
        """
        if data_frame_labels is not None:
            return tf.data.Dataset.from_tensor_slices((dict(data_frame_features), data_frame_labels.values))
        return tf.data.Dataset.from_tensor_slices(dict(data_frame_features))

    def map(self, data_frame_features=None, data_frame_labels=None):
        """map pandas dataframe to tf.data.Dataset

        Args:
            data_frame_features (pandas.DataFrame): Features Data.
            data_frame_labels (pandas.DataFrame): Target Data
        """
        return self.data_frame_to_dataset(data_frame_features, data_frame_labels)

    def split_data_set(self, dataset=None, val_split_fraction=0.25):
        """Split data for training and validation

        Args:
            features (tf.data.Dataset): Dataset to split into training and validation
        
        Returns:
            (tf.data.Dataset, tf.data.Dataset): train and validation datasets
        """
        rows = dataset.cardinality().numpy()
        dataset = dataset.shuffle(rows)
        training_size = int((1 - val_split_fraction) * rows)
        train_data_set = dataset.take(training_size)
        val_data_set = dataset.skip(training_size)
        return train_data_set, val_data_set
