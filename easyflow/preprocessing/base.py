"""base classes for stateful preprocessing layers"""
from abc import ABC, abstractmethod

import tensorflow as tf


def extract_feature_column(dataset, name):
    dataset = dataset.map(lambda x, y: x[name])
    dataset = dataset.map(lambda x: tf.expand_dims(x, -1))
    return dataset


class BaseFeatureTransformer:
    """Apply column based transformation on the data

    Args:
        feature_encoder_list : 
    """
    def __init__(self, feature_encoder_list=None):
        self.feature_encoder_list = feature_encoder_list

    def create_inputs(self, features, dtype):
        """Create inputs for Keras Model

        Returns:
            list: list of keras inputs
        """
        return [tf.keras.Input(shape=(), name=feature, dtype=dtype) for feature in features]

    @abstractmethod
    def transform(self, dataset):
        """Apply feature encodings on supplied list

        Args:
            X (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, list): Keras inputs for each feature and list of encoders
        """

    def __len__(self):
        return len(self.feature_encoder_list)
