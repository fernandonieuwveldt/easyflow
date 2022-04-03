"""Main feature preprocessing interfaces"""

import tensorflow as tf
from easyflow.preprocessing.feature_preprocessor_layer import (
    BaseFeaturePreprocessorLayer,
    BaseFeaturePreprocessorFromTensorflowDataset
)


FeaturePreprocessor = BaseFeaturePreprocessorLayer


class FeaturePreprocessorUnion(tf.keras.layers.Layer):
    """Apply column based preprocessing on the data and combine features with a concat layer.

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """

    def __init__(self, feature_preprocessor_list, *args, **kwargs):
        super(FeaturePreprocessorUnion, self).__init__(*args, **kwargs)
        self.feature_preprocessor_list = feature_preprocessor_list
        self.preprocessor_flow = BaseFeaturePreprocessorFromTensorflowDataset(
                self.feature_preprocessor_list
        )

    @classmethod
    def from_infered_pipeline(cls, dataset):
        """Infer standard pipeline for structured data, i.e Normalization for numerical
        features and StringLookup/IntegerLookup for categoric features

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            BaseFeaturePreprocessorLayer: Initilized BaseFeaturePreprocessorLayer object
        """
        if isinstance(dataset, tf.data.Dataset):
            feature_preprocessor_list = BaseFeaturePreprocessorFromTensorflowDataset.from_infered_pipeline(dataset)
            return cls(feature_preprocessor_list)

    def adapt(self, dataset):
        """Adapt preprocessing layers.

        Args:
            dataset ([pandas.DataFrame, tf.data.Dataset]): Training data.
        """
        if isinstance(dataset, tf.data.Dataset):
            # self.preprocessor_flow = BaseFeaturePreprocessorFromTensorflowDataset(
            #     self.feature_preprocessor_list
            # )
            self.preprocessor_flow.adapt(dataset)

    @tf.function
    def call(self, inputs):
        """Join features. If more flexibility and customization is needed use PreprocessorColumnTransformer.

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, tf.keras.layer): Keras inputs for each feature and concatenated layer
        """
        preprocessed_input = self.preprocessor_flow(inputs)
        # flatten (or taking the union) of preprocessed inputs
        preprocessed_input = [
            low_level
            for _, high_level in preprocessed_input.items()
            for low_level in high_level
        ]
        if len(preprocessed_input) > 1:
            return tf.keras.layers.concatenate(preprocessed_input)
        return preprocessed_input.pop()


# class FeaturePreprocessorUnion(BaseFeaturePreprocessorLayer):
#     """Apply column based preprocessing on the data and combine features with a concat layer.

#     Args:
#         feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
#     """

#     def __init__(self, *args, **kwargs):
#         super(FeaturePreprocessorUnion, self).__init__(*args, **kwargs)

#     def call(self, inputs):
#         """Join features. If more flexibility and customization is needed use PreprocessorColumnTransformer.

#         Args:
#             dataset (tf.data.Dataset): Features Data to apply encoder on.

#         Returns:
#             (list, tf.keras.layer): Keras inputs for each feature and concatenated layer
#         """
#         preprocessed_input = super(FeaturePreprocessorUnion, self).call(inputs)
#         # flatten (or taking the union) of preprocessed inputs
#         preprocessed_input = [
#             low_level
#             for _, high_level in preprocessed_input.items()
#             for low_level in high_level
#         ]
#         if len(preprocessed_input) > 1:
#             return tf.keras.layers.concatenate(preprocessed_input)
#         return preprocessed_input.pop()
