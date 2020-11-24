"""
data loading
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
            data_frame ([type], optional): [description]. Defaults to None.
            labels ([type], optional): [description]. Defaults to None.
        """
        dataset = tf.data.Dataset.from_tensor_slices((dict(data_frame_features), data_frame_labels.values)).shuffle(1000)
        return dataset       

    def transform(self, data_frame_features=None, data_frame_labels=None):
        """transform data

        Args:
            data_frame_features ([type], optional): [description]. Defaults to None.
            data_frame_labels ([type], optional): [description]. Defaults to None.
        """
        return self.data_frame_to_dataset(data_frame_features, data_frame_labels)
