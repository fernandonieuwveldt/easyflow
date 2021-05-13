"""Main feature preprocessing interfaces"""

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import PreprocessingLayer

from .base import BaseEncoder
from .base import extract_feature_column


class Encoder(BaseEncoder):
    """
    Preprocess each feature based on specified preprocessing layer contained in feature_encoder_list
    """
    def __init__(self, feature_encoder_list=None):
        super().__init__(feature_encoder_list=feature_encoder_list)

    def encode(self, dataset):
        """Apply feature encodings on supplied feature encoding list

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, dict): Keras inputs for each feature and dict of encoders
        """
        name, preprocessor, features = self.feature_encoder_list
        feature_inputs = self.create_inputs(features, preprocessor.dtype)
        encoded_features = self._encode_one(dataset, preprocessor, features, feature_inputs)
        return feature_inputs, encoded_features


class SequentialEncoder(BaseEncoder):
    """
    Preprocessing pipeline to apply multiple encoders in serie
    """
    def __init__(self, feature_encoder_list=None):
        super().__init__(feature_encoder_list=feature_encoder_list)
        self.feature_encoder_list = self.map_steps(self.feature_encoder_list)
    
    def map_steps(self, steps=None):
        name, preprocessors, features = self.feature_encoder_list
        if not isinstance(preprocessors, list):
            return self.feature_encoder_list

        steps_mapped = []
        for preprocessor in preprocessors:
            steps_mapped.append((name, preprocessor, features))
        return steps_mapped

    def encode(self, dataset):
        """Apply feature encodings on supplied feature encoding list

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, dict): Keras inputs for each feature and dict of encoders
        """
        name, preprocessor, features = self.feature_encoder_list[0]
        feature_inputs = self.create_inputs(features, preprocessor.dtype)
        # TODO: feature_inputs and encoded_features should be of the same type
        encoded_features = self._encode_one(dataset, preprocessor, features, feature_inputs)
        if len(self.feature_encoder_list) == 1:
            # SequentialEncoder use case is for multiple encoders applied on the same features
            # It should never have only one encoder. Adding this step for robustness
            return feature_inputs, encoded_features
        for (name, preprocessor, features) in self.feature_encoder_list[1:]:
            encoded_features = self._encode_one(dataset, preprocessor, features, [v for v in encoded_features.values()])
        return feature_inputs, encoded_features


class Pipeline:
    """
    Main interface for transforming features. Apply feature encoder list which can contain both
    Encoder and SequentialEncoder object types

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """
    def __init__(self, feature_encoder_list=None):
        self.feature_encoder_list = feature_encoder_list

    def encode(self, dataset):
        """Apply feature encodings on supplied feature encoding list

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, dict): Keras inputs for each feature and dict of encoders
        """
        all_feature_inputs, all_feature_encoders = [], {}
        for step in self.feature_encoder_list:
            if isinstance(step[1], list):
                encoder = SequentialEncoder(step)
            else:
                encoder = Encoder(step)
            encoder_step_name = encoder.encoder_name
            feature_inputs, feature_encoders = encoder.encode(dataset)
            all_feature_inputs.extend(feature_inputs)
            all_feature_encoders[encoder_step_name] = feature_encoders
        return all_feature_inputs, all_feature_encoders

class FeatureUnion(Pipeline):
    """Apply column based preprocessing on the data and combine features with a concat layer.

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, dataset):
        """Join features. If more flexibility and customization is needed use PreprocessorColumnTransformer.

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, tf.keras.layer): Keras inputs for each feature and concatenated layer
        """
        feature_layer_inputs, feature_encoders = super(FeatureUnion, self).encode(dataset)
        # flatten (or taking the union) of feature encoders
        feature_encoders = [low_level for high_level in feature_encoders.values()\
                                        for low_level in high_level.values()]
        if len(feature_encoders) > 1:
            return feature_layer_inputs, tf.keras.layers.concatenate(feature_encoders)
        return feature_layer_inputs, feature_encoders.pop()
