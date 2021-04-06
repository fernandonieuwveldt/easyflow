"""
Transformer for tensorflow feature columns
"""
from .feature_encoder import NumericalFeatureEncoder, CategoricalFeatureEncoder


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
        feature_encoders = {}
        for (name, encoder, features) in self.feature_encoder_list:
            feature_inputs, feature_encoded = encoder.encode(dataset, features)
            feature_layer_inputs.update(feature_inputs)
            feature_encoders[name] = feature_encoded
        return feature_layer_inputs, feature_encoders


class FeatureUnionTransformer(FeatureColumnTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, X):
        """Join features. If more flexibility and customization is needed use FeatureColumnTransformer.

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        feature_layer_inputs, feature_encoders = super().transform(X)
        # flatten (or taking the union) of feature encoders 
        return feature_layer_inputs, [fe for transformer in list(feature_encoders.values())\
            for fe in transformer]
