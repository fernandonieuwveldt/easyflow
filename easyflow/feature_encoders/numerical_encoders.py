"""
Interface to use tensorflow feature columns with Keras for Numerical features
"""
import tensorflow as tf

from .base import BaseFeatureColumnEncoder


class NumericalFeatureEncoder(BaseFeatureColumnEncoder):
    """
    Class encodes numerical features using tensorflow feature_columns
    """
    def __init__(self):
        pass

    def encode(self, dataset=None, features=None):
        """Encoding numerical type features with tensorflow feature columns

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        numerical_inputs, feature_encoders = {}, {}

        for feature in features:
            numerical_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
            feature_encoders[feature] = tf.feature_column.numeric_column(feature)
        return numerical_inputs, [feature for _, feature in feature_encoders.items()]
