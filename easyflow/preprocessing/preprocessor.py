import tensorflow as tf

from .base import BaseFeatureTransformer
from .base import extract_feature_column


class PipelineFeatureTransformer(BaseFeatureTransformer):
    """
    Preprocessing pipeline to apply multiple encoders in serie
    """
    def _warm_up(self, dataset, preprocessor, features, feature_inputs):
        """Apply feature encodings on supplied list

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        # repeatable code for warm start to avoid using conditions in the Graph
        adapted_preprocessors = {}
        encoded_features = {}
        for feature_input, feature_name in zip(feature_inputs, features):
            _preprocessor = preprocessor()
            feature_ds = extract_feature_column(dataset, feature_name)
            _preprocessor.adapt(feature_ds)
            encoded_feature = _preprocessor(feature_input)
            encoded_features[feature_name] = encoded_feature
            adapted_preprocessors[feature_name] = _preprocessor
        return adapted_preprocessors, encoded_features

    def transform(self, dataset):
        """Apply feature encodings on supplied list

        Args:
            X (tf.data.DataF): Features Data to apply encoder on.

        Returns:
            (list, list): Keras inputs for each feature and list of encoders
        """
        name, preprocessor, features, dtype = self.feature_encoder_list[0]
        feature_inputs = self.create_inputs(features, dtype)
        adapted_preprocessors, encoded_features = self._warm_up(dataset, preprocessor, features, feature_inputs)
        feature_layer_inputs = {}
        for (name, preprocessor, features, dtype) in self.feature_encoder_list[1:]:
            for feature_name in features:
                _preprocessor = preprocessor()
                feature_ds = extract_feature_column(dataset, feature_name)
                feature_ds = feature_ds.map(adapted_preprocessors[feature_name])
                _preprocessor.adapt(feature_ds)
                encoded_feature = _preprocessor(encoded_features[feature_name])
                adapted_preprocessors[feature_name] = _preprocessor
                encoded_features[feature_name] = encoded_feature
        return feature_inputs, encoded_features


class FeatureTransformer(BaseFeatureTransformer):

    def transform(self, dataset):
        """Apply feature encodings on supplied list

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        feature_layer_inputs = []
        feature_encoders = {}
        for (name, preprocessor, features, dtype) in self.feature_encoder_list:
            feature_inputs = self.create_inputs(features, dtype)
            for k, feature_name in enumerate(features):
                _preprocessor = preprocessor()
                feature_ds = extract_feature_column(dataset, feature_name)
                _preprocessor.adapt(feature_ds)
                encoded_feature = _preprocessor(feature_inputs[k])
                feature_encoders[feature_name] = encoded_feature
            feature_layer_inputs.extend(feature_inputs)
        return feature_layer_inputs, feature_encoders


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
        return all_feature_inputs, all_feature_encoders


class FeatureUnionTransformer(FeatureTransformer):
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
