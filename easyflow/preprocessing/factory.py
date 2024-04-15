"""This class will serve as a factory for different data types"""

import pandas as pd
import tensorflow as tf
from easyflow.preprocessing.feature_preprocessor_layer import (
    FeaturePreprocessorFromTensorflowDataset,
    FeaturePreprocessorFromPandasDataFrame,
)


class FeaturePreprocessorFactory(tf.keras.models.Model):
    """Apply column based transformation on the data using tf.keras  preprocessing layers.

    Args:
        feature_encoder_list : List of preprocessor of the form: ('name', preprocessor type, list of features)
    """

    def __init__(self, feature_preprocessor_list=[], *args, **kwargs):
        super(FeaturePreprocessorFactory, self).__init__(*args, *kwargs)
        self.feature_preprocessor_list = feature_preprocessor_list
        self.preprocessor_flow = None

    @classmethod
    def from_infered_pipeline(cls, dataset):
        """Infer standard pipeline for structured data, i.e Normalization for numerical
        features and StringLookup/IntegerLookup for categoric features

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            FeaturePreprocessorFactory: Initilized FeaturePreprocessorFactory object
        """
        if isinstance(dataset, tf.data.Dataset):
            feature_preprocessor_list = (
                FeaturePreprocessorFromTensorflowDataset.from_infered_pipeline(dataset)
            )
            return cls(feature_preprocessor_list)

    def adapt(self, dataset):
        """Adapt preprocessing layers.

        Args:
            dataset ([pandas.DataFrame, tf.data.Dataset]): Training data.
        """
        if isinstance(dataset, tf.data.Dataset):
            self.preprocessor_flow = FeaturePreprocessorFromTensorflowDataset(
                self.feature_preprocessor_list
            )

        if isinstance(dataset, pd.DataFrame):
            self.preprocessor_flow = FeaturePreprocessorFromPandasDataFrame(
                self.feature_preprocessor_list
            )

        if not self.preprocessor_flow:
            raise Exception("Datatype not supported: dataset should be of type pandas DataFrame or Tensorflow tf.data.Dataset")

        self.preprocessor_flow.adapt(dataset)

    def call(self, inputs):
        """Apply adapted layers on new data

        Args:
            inputs (dict): Dictionary of Tensors.

        Returns:
            dict: Dict of Tensors
        """
        if not self.preprocessor_flow:
            raise Exception("First run adapt before forward pass.")

        return self.preprocessor_flow(inputs)
