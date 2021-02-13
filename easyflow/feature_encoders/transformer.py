"""
Transformer for tensorflow feature columns
"""
from .feature_encoder import NumericalFeatureEncoder, CategoricalFeatureEncoder


class FeatureColumnTransformer:
    """Apply column based transformation on the data

    Args:
        feature_encoder_list : 
    """
    def __init__(self, feature_encoder_list=None):
        self.feature_encoder_list = feature_encoder_list
        
    def transform(self, X):
        """Apply feature encodings on supplied list

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        feature_layer_inputs = {}
        feature_encoders = {}
        for (name, encoder, features) in self.feature_encoder_list:
            feature_inputs, feature_encoded = encoder.encode(X, features)
            feature_layer_inputs.update(feature_inputs)
            feature_encoders.update(feature_encoded)
        return feature_layer_inputs, feature_encoders


if __name__ == '__main__':
    """
    An example:

    feature_encoder_list = [('numeric_encoder', NumericalFeatureEncoder(), NUMERIC_FEATURE_COLUMNS),
                            ('categoric_encoder', CategoricalFeatureEncoder(), CATEGORICAL_FEATURE_COLUMNS)
                            ]
    transformer = FeatureColumnTransformer(feature_encoder_list)
    transformer.encode(data)
    """
