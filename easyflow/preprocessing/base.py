"""base classes for stateful preprocessing layers"""
from abc import ABC, abstractmethod
import tensorflow as tf

from .custom import IdentityPreprocessingLayer


def one2one_func(x):
    """helper method to apply one to one preprocessor"""
    return x


def extract_feature_column(dataset, name):
    feature = dataset.map(lambda x, y: x[name])
    feature = feature.map(lambda x: tf.expand_dims(x, -1))
    return feature


class BaseEncoder:
    """Apply column based transformation on the data

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """
    def __init__(self, feature_encoder_list=None):
        self.feature_encoder_list = feature_encoder_list
        self.check_and_map()
        features = self.feature_encoder_list[0][2]
        self.adapted_preprocessors = {feature_name: one2one_func for feature_name in features}

    @abstractmethod
    def encode(self, dataset):
        """Apply feature encodings on supplied feature encoding list

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, dict): Keras inputs for each feature and dict of encoders
        """

    def check_and_map(self):
        """Check and Map input if any of the preprocessors are None, i.e. use as is
        """
        for k, (name, preprocessor, features) in enumerate(self.feature_encoder_list):
            self.feature_encoder_list[k] = (name, preprocessor or IdentityPreprocessingLayer(), features)

    def create_inputs(self, features, dtype):
        """Create list of keras Inputs

        Returns:
            list: list of keras inputs
        """
        return [tf.keras.Input(shape=(), name=feature, dtype=dtype) for feature in features]

    def _encode_one(self, dataset, preprocessor, features, feature_inputs):
        """Apply feature encodings on supplied list

        Args:
            X (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, list): Keras inputs for each feature and list of encoders
        """
        encoded_features = {}
        # get initial preprocessing layer config 
        config = preprocessor.get_config()
        for k, (feature_input, feature_name) in enumerate(zip(feature_inputs, features)):
            config.pop('name', None)
            _preprocessor = preprocessor if k==0 else preprocessor.from_config(config)
            feature_ds = extract_feature_column(dataset, feature_name)
            feature_ds = feature_ds.map(self.adapted_preprocessors[feature_name])
            _preprocessor.adapt(feature_ds)
            encoded_feature = _preprocessor(feature_input)
            encoded_features[feature_name] = encoded_feature
            self.adapted_preprocessors[feature_name] = _preprocessor
        return encoded_features

    def __len__(self):
        return len(self.feature_encoder_list)
