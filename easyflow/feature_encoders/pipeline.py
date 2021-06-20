"""
Transformer for tensorflow feature columns
"""

import tensorflow as tf


class FeatureColumnTransformer:
    """Apply column based transformation on the data

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """
    def __init__(self, feature_encoder_list=None):
        self.feature_encoder_list = feature_encoder_list
        
    def transform(self, dataset):
        """Apply feature encodings on supplied list

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        feature_layer_inputs = {}
        feature_encoder_layer = {}
        for (name, encoder, features) in self.feature_encoder_list:
            # best to use create_inputs and encode independently?
            feature_inputs, feature_encoded = encoder.transform(dataset, features)
            feature_layer_inputs.update(feature_inputs)
            feature_encoder_layer[name] = feature_encoded
        # Apply DenseFeatures layer when all feature layer inputs are available
        for name, encoded_feature in feature_encoder_layer.items():
            feature_encoder_layer[name] = tf.keras.layers.DenseFeatures(encoded_feature)(feature_layer_inputs)
        return feature_layer_inputs, feature_encoder_layer


class FeatureUnionTransformer(FeatureColumnTransformer):
    """Apply transformations and apply union/concatenating individual feature layers
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, dataset):
        """Join features. If more flexibility and customization is needed use FeatureColumnTransformer.

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        feature_layer_inputs, feature_encoder_layer = super().transform(dataset)
        feature_layer = [feature_layer for feature_layer in feature_encoder_layer.values()]
        if len(feature_layer) == 1:
            return feature_layer_inputs, feature_layer.pop()
        return feature_layer_inputs, tf.keras.layers.concatenate(copy.copy(feature_layer))
