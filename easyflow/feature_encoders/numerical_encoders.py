"""
Interface to use tensorflow feature columns with Keras for Numerical features
"""

import tensorflow as tf

from .base import BaseFeatureColumnEncoder


class NumericalFeatureEncoder(BaseFeatureColumnEncoder):
    """
    Class encodes numerical features using tensorflow feature_columns
    """
    def __init__(self, **kwargs):
        super().__init__(feature_transformer=tf.feature_column.numeric_column, **kwargs)


class BucketizedFeatureEncoder(BaseFeatureColumnEncoder):
    """Class applies bucketized encoding by first applying NumericalFeatureEncoder
    """
    def __init__(self, **kwargs):
        super().__init__(feature_transformer=tf.feature_column.bucketized_column, **kwargs)

    def encode(self, dataset=None, features=None):
        """Apply bucketized feature encoding by first applying numerical encoder.

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (list): list of encoded features
        """
        numerical_encoder = NumericalFeatureEncoder()
        numerical_inputs, numerical_encoded_features = numerical_encoder.transform(dataset, features)
        return [self.feature_transformer(feature, **self.kwargs) for feature in numerical_encoded_features]
