import tensorflow as tf

from .base import BaseFeatureTransformer
from .base import extract_feature_column


class FeatureTransformer(BaseFeatureTransformer):

    def transform(self, dataset):
        """Apply feature encodings on supplied list

        Args:
            X (tf.data.DataF): Features Data to apply encoder on.

        Returns:
            (list, list): Keras inputs for each feature and list of encoders
        """
        name, preprocessor, features, dtype = self.feature_encoder_list[0]
        feature_layer_inputs = []
        feature_encoders = {}
        for (name, preprocessor, features, dtype) in self.feature_encoder_list:
            feature_inputs = self.create_inputs(features, dtype)
            encoded_features = self._warm_up(dataset, preprocessor, features, feature_inputs)
            feature_layer_inputs.extend(feature_inputs)
            feature_encoders.update(encoded_features)
        return feature_layer_inputs, feature_encoders


class PipelineFeatureTransformer(BaseFeatureTransformer):
    """
    Preprocessing pipeline to apply multiple encoders in serie
    """
    def transform(self, dataset):
        """Apply feature encodings on supplied list

        Args:
            X (tf.data.DataF): Features Data to apply encoder on.

        Returns:
            (list, list): Keras inputs for each feature and list of encoders
        """
        name, preprocessor, features, dtype = self.feature_encoder_list[0]
        feature_inputs = self.create_inputs(features, dtype)
        # TODO: feature_inputs and encoded_features should be of the same type
        encoded_features = self._warm_up(dataset, preprocessor, features, feature_inputs)
        for (name, preprocessor, features, dtype) in self.feature_encoder_list[1:]:
            encoded_features = self._warm_up(dataset, preprocessor, features, [v for v in encoded_features.values()])
        return feature_inputs, encoded_features


class Transformer:
    """
    Main interface for transforming features.
    """
    def __init__(self, feature_encoder_list=None):
        self.feature_encoder_list = feature_encoder_list

    def transform(self, dataset):
        """
        Apply feature encoder list.
        """
        all_feature_inputs, all_feature_encoders = [], {}
        for step in self.feature_encoder_list:
            feature_inputs, feature_encoders = step.transform(dataset)
            all_feature_inputs.extend(feature_inputs)
            all_feature_encoders.update(feature_encoders)
        return all_feature_inputs, [fe for fe in all_feature_encoders.values()]


class UnionTransformer(Transformer):
    """Apply column based preprocessing on the data

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, X):
        """Join features. If more flexibility and customization is needed use PreprocessorColumnTransformer.

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.

        Returns:
            (dict, tf.keras.layer): Keras inputs for each feature and concatenated layer
        """
        feature_layer_inputs, feature_encoders = super().transform(X)
        # flatten (or taking the union) of feature encoders 
        return feature_layer_inputs, tf.keras.layers.concatenate(feature_encoders)
