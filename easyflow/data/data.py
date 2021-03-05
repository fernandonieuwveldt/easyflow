"""
data loading and transforming pandas dataframe to Tensorflow Dataset
"""

import pandas
import tensorflow as tf


class TFDataTransformer:
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
        total_samples = min(1024, len(data_frame_features))
        if data_frame_labels is not None:
            return tf.data.Dataset.from_tensor_slices((dict(data_frame_features), data_frame_labels.values)).shuffle(total_samples)
        else:
            return tf.data.Dataset.from_tensor_slices(dict(data_frame_features))

    def transform(self, data_frame_features=None, data_frame_labels=None):
        """transform data

        Args:
            data_frame_features (pandas.DataFrame): Features Data.
            data_frame_labels (pandas.DataFrame): Target Data
        """
        return self.data_frame_to_dataset(data_frame_features, data_frame_labels)
