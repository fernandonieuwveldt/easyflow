"""Main feature preprocessing interfaces"""

import tensorflow as tf
from easyflow.preprocessing.factory import (
    FeaturePreprocessorFactory,
)


class FeaturePreprocessor(FeaturePreprocessorFactory):
    """
    Main interface for transforming features. Apply feature preprocessing list which can contain both
    Encoder and SequentialEncoder object types
    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FeatureUnion(FeaturePreprocessorFactory):
    """Apply column based preprocessing on the data and combine features with a concat layer.

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """

    def __init__(self, *args, **kwargs):
        super(FeatureUnion, self).__init__(*args, **kwargs)

    def call(self, inputs):
        """Join features. If more flexibility and customization is needed use PreprocessorColumnTransformer.

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, tf.keras.layer): Keras inputs for each feature and concatenated layer
        """
        preprocessed_input = super(FeatureUnion, self).call(inputs)
        # flatten (or taking the union) of preprocessed inputs
        preprocessed_input = [
            low_level
            for _, high_level in preprocessed_input.items()
            for low_level in high_level
        ]
        if len(preprocessed_input) > 1:
            return tf.keras.layers.concatenate(preprocessed_input)
        return preprocessed_input.pop()
