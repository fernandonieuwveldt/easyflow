"""Base Classes for encoders using tensrflow feature columns"""

from abc import ABC, abstractmethod

import tensorflow as tf


class BaseFeatureColumnEncoder(ABC):
    """Base class for a tensorlow feature column based encoder"""

    def __init__(self, feature_transformer=None, **kwargs):
        self.feature_transformer = feature_transformer
        self.kwargs = kwargs

    def create_inputs(self, dataset=None, features=None):
        """Create model inputs

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict): Keras inputs for each feature
        """
        def get_data_type(ds, feature):
            """helper function to get dtype of feature"""
            if len(ds.element_spec) == 2:
                return ds._structure[0][feature].dtype
            return ds._structure[feature].dtype

        return {feature: tf.keras.Input(shape=(1,), name=feature, dtype=get_data_type(dataset, feature))\
            for feature in features}

    # should perhaps be a abstractmethod
    def encode(self, dataset=None, features=None):
        """Apply feature encoding. This method can be over ridden for specific use cases

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (list): list of encoded features
        """
        return [self.feature_transformer(feature, **self.kwargs) for feature in features]

    def transform(self, dataset=None, features=None):
        """Encoding numerical type features with tensorflow feature columns

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        return self.create_inputs(dataset, features), self.encode(dataset, features)
