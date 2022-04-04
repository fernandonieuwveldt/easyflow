"""base classes for stateful preprocessing layers"""
import tensorflow as tf
from tensorflow.keras import layers

from easyflow.preprocessing.custom import SequentialPreprocessingChainer
from easyflow.preprocessing.base import BaseFeaturePreprocessor


class FeaturePreprocessorFromTensorflowDataset(tf.keras.layers.Layer, BaseFeaturePreprocessor):
    """Feature Layer for Tensorflow Dataset type
    """

    def __init__(self, feature_preprocessor_list=[], *args, **kwargs):
        super(FeaturePreprocessorFromTensorflowDataset, self).__init__(*args, *kwargs)
        self.feature_preprocessor_list = feature_preprocessor_list
        self.feature_preprocessor_list = self.map_preprocessor(self.feature_preprocessor_list)
        self.adapted_preprocessors = dict()

    def adapt(self, dataset):
        """Adapt layers from tf.data.Dataset source type.

        Args:
            dataset (tf.data.Dataset): Training data.
        """
        for _, preprocessor, features in self.feature_preprocessor_list:
            # get initial preprocessing layer config
            config = preprocessor.get_config()
            for k, feature in enumerate(features):
                # get a fresh preprocessing instance
                cloned_preprocessor = preprocessor if k==0 else preprocessor.from_config(config)
                feature_ds = extract_feature_column(dataset, feature)
                # check if layer has adapt method
                cloned_preprocessor.adapt(feature_ds)
                self.adapted_preprocessors[feature] = cloned_preprocessor

    @tf.function
    def call(self, inputs):
        """Apply adapted layers on new data

        Args:
            inputs (dict): Dictionary of Tensors.

        Returns:
            dict: Dict of Tensors
        """
        forward_pass = dict()
        for name, _, features in self.feature_preprocessor_list:
            forward_pass[name] = [
                self.adapted_preprocessors[feature](inputs[feature])
                for feature in features
            ]
        return forward_pass

    @classmethod
    def from_infered_pipeline(cls, dataset):
        """Infer preprocessing spec from tf.data.Dataset type

        Args:
            dataset (tf.data.Dataset): Training data that contains features and/or target

        Returns:
            list: steps for preprocessing list
        """
        numeric_features = []
        categoric_features = []
        string_categoric_features = []
        for feature, _type in dataset.element_spec[0].items():
            if _type.dtype == tf.string:
                string_categoric_features.append(feature)
            elif _type.dtype == tf.int64:
                categoric_features.append(feature)
            else:
                numeric_features.append(feature)

        feature_preprocessor_list = [
            # FIXME: We will most likely not have all these steps
            (
                "numerical_features", layers.Normalization(), numeric_features
            ),
            (
                "categorical_features", layers.IntegerLookup(output_mode="binary"), categoric_features,
            ),
            (
                "string_categorical_features",
                SequentialPreprocessingChainer([layers.StringLookup(), layers.IntegerLookup(output_mode="binary")]),
                string_categoric_features,
            ),
        ]
        return feature_preprocessor_list


class FeaturePreprocessorFromPandasDataFrame(tf.keras.layers.Layer, BaseFeaturePreprocessor):
    """Feature Layer flow for pandas DataFrame source type
    """

    def __init__(self, feature_preprocessor_list=[], *args, **kwargs):
        super(FeaturePreprocessorFromPandasDataFrame, self).__init__(*args, *kwargs)
        self.feature_preprocessor_list = self.map_preprocessor(feature_preprocessor_list)
        self.adapted_preprocessors = dict()


def extract_feature_column(dataset, name):
    """Extract feature from dataset

    Args:
        dataset (tf.data.Dataset): Training Data
        name (str): Name of feature to extract

    Returns:
        tf.data.Dataset: Mapped dataset for supplied feature.
    """
    feature = dataset.map(lambda x, y: x[name])
    feature = feature.map(lambda x: tf.expand_dims(x, -1))
    return feature
