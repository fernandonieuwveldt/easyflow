"""Main feature preprocessing interfaces"""

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization, StringLookup, IntegerLookup

from .base import _BaseSingleEncoder, _BaseMultipleEncoder


class Pipeline:
    """
    Main interface for transforming features. Apply feature encoder list which can contain both
    Encoder and SequentialEncoder object types

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """
    def __init__(self, feature_encoder_list=None):
        self.feature_encoder_list = feature_encoder_list

    @classmethod
    def from_infered_pipeline(cls, dataset):
        """Infer standard pipeline for structured data, i.e NumericalFeatureEncoder for numeric
        features and CategoricalFeatureEncoder for categoric features

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list): basic encoding list
        """
        numeric_features = []
        categoric_features = []
        string_categoric_features = []
        # change loop over dtypes such that encoders can be created in flight
        for feature, _type in dataset.element_spec[0].items():
            if _type.dtype == tf.string:
                string_categoric_features.append(feature)
            elif _type.dtype == tf.int64:
                categoric_features.append(feature)
            else:
                numeric_features.append(feature)

        encoding_list = [('numerical_features', Normalization(), numeric_features),
                         ('categorical_features', IntegerLookup(output_mode='binary'), categoric_features),
                         ('string_categorical_features', [StringLookup(), IntegerLookup(output_mode='binary')], string_categoric_features)]

        return cls(encoding_list)

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
                encoder = _BaseMultipleEncoder(step)
            else:
                encoder = _BaseSingleEncoder(step)
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
