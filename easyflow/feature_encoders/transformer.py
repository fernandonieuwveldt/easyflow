"""
Transformer for tensorflow feature columns
"""
from .feature_encoder import NumericalFeatureEncoder, CategoricalFeatureEncoder


class FeatureColumnTransformer:
    """[summary]
    """
    def __init__(self, feature_encoder_list=None):
        """Transformer for feature columns

        Args:
            feature_encoder_list : 
        """
        self.feature_encoder_list = feature_encoder_list
        
    def transform(self, X):
        """
        """

        feature_encoders = []
        feature_layer_inputs = {}
        for (name, encoder, features) in self.feature_encoder_list:
            feature_inputs, feature_encoded = encoder.encode(X, features)
            # perhaps feature_encoders should be dictionaries as well?
            feature_encoders.append(feature_encoded)
            feature_layer_inputs.update(feature_inputs)
        return feature_layer_inputs, feature_encoders


if __name__ == '__main__':
    feature_encoder_list = [('numeric_encoder', NumericalFeatureEncoder(), NUMERIC_FEATURE_COLUMNS),
                            ('categoric_encoder', CategoricalFeatureEncoder(), CATEGORICAL_FEATURE_COLUMNS)
                            ]
    transformer = FeatureColumnTransformer(feature_encoder_list)
    transformer.encode()
