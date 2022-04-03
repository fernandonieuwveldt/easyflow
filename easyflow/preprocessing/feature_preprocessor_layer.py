"""base classes for stateful preprocessing layers"""
import tensorflow as tf
from tensorflow.keras import layers

from easyflow.preprocessing.custom import SequentialPreprocessingChainer
from easyflow.preprocessing.base import BaseFeaturePreprocessor


class BaseFeaturePreprocessorLayer(tf.keras.layers.Layer):
    """Apply column based transformation on the data using tf.keras  preprocessing layers.

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """

    def __init__(self, feature_preprocessor_list=[], *args, **kwargs):
        super(BaseFeaturePreprocessorLayer, self).__init__(*args, *kwargs)
        self.feature_preprocessor_list = feature_preprocessor_list

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
            self.preprocessor_flow = BaseFeaturePreprocessorFromTensorflowDataset(
                self.feature_preprocessor_list
            )
            self.preprocessor_flow.adapt(dataset)

    def call(self, inputs):
        """Apply adapted layers on new data

        Args:
            inputs (dict): Dictionary of Tensors.

        Returns:
            dict: Dict of Tensors
        """
        return self.preprocessor_flow(inputs)


class BaseFeaturePreprocessorFromTensorflowDataset(tf.keras.layers.Layer):
    """Feature Layer for Tensorflow Dataset type
    """

    def __init__(self, feature_preprocessor_list=[], *args, **kwargs):
        super(BaseFeaturePreprocessorFromTensorflowDataset, self).__init__(*args, *kwargs)
        self.feature_preprocessor_list = feature_preprocessor_list
        # self.feature_preprocessor_list = self.map_preprocessor(self.feature_preprocessor_list)
        self.adapted_preprocessors = {}

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
        forward_pass = {}
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


class BaseFeaturePreprocessorFromPandasDataFrame(tf.keras.layers.Layer, BaseFeaturePreprocessor):
    """Feature Layer flow for pandas DataFrame source type
    """

    def __init__(self, feature_preprocessor_list=[], *args, **kwargs):
        super(BaseFeaturePreprocessorFromPandasDataFrame, self).__init__(*args, *kwargs)
        self.feature_preprocessor_list = self.map_preprocessor(feature_preprocessor_list)
        self.adapted_preprocessors = {}


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
