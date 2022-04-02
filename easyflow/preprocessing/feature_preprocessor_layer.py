"""base classes for stateful preprocessing layers"""
import tensorflow as tf
from tensorflow.keras import layers

from easyflow.preprocessing.custom import (
    NumericPreprocessingLayer,
    PreprocessingChainer,
)


def one2one_func(x):
    """helper method to apply one to one preprocessor"""
    return x


def extract_feature_column(dataset, name):
    feature = dataset.map(lambda x, y: x[name])
    feature = feature.map(lambda x: tf.expand_dims(x, -1))
    return feature


class BaseFeaturePreprocessorLayer(tf.keras.layers.Layer):
    """Apply column based transformation on the data using tf.keras  preprocessing layers.

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """

    def __init__(self, feature_preprocessor_list=[], *args, **kwargs):
        super(BaseFeaturePreprocessorLayer, self).__init__(*args, *kwargs)
        feature_preprocessor_list = self.map_preprocessor(feature_preprocessor_list)
        self.feature_preprocessor_list = feature_preprocessor_list
        self.adapted_preprocessors = {}

    @classmethod
    def from_infered_pipeline(cls, dataset):
        """Infer standard pipeline for structured data, i.e NumericalFeatureEncoder for numeric
        features and CategoricalFeatureEncoder for categoric features

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list): basic encoding list
        """
        if isinstance(dataset, tf.data.Dataset):
            feature_preprocessor_list = cls._infer_from_tf_data(dataset)
            return cls(feature_preprocessor_list)

    @staticmethod
    def _infer_from_tf_data(dataset):
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
            ("numerical_features", layers.Normalization(), numeric_features),
            (
                "categorical_features",
                layers.IntegerLookup(output_mode="binary"),
                categoric_features,
            ),
            (
                "string_categorical_features",
                PreprocessingChainer(
                    [layers.StringLookup(), layers.IntegerLookup(output_mode="binary")]
                ),
                string_categoric_features,
            ),
        ]
        return feature_preprocessor_list

    @staticmethod
    def _infer_from_pandas_data_frame(dataset):
        return []

    def map_preprocessor(self, steps):
        """Check and Map input if any of the preprocessors are None, i.e. use as is. For 
        example Binary features that don't need further preprocessing        
        """
        selector = lambda _preprocessor: _preprocessor or NumericPreprocessingLayer()
        return [
            (name, selector(preprocessor), step) for name, preprocessor, step in steps
        ]

    def adapt_tf_dataset(self, dataset):
        for _, preproc_steps, features in self.feature_preprocessor_list:
            # get initial preprocessing layer config
            config = preproc_steps.get_config()
            for feature in features:
                # get a fresh preprocessing instance
                cloned_preprocessor = preproc_steps.from_config(config)
                feature_ds = extract_feature_column(dataset, feature)
                # check if layer has adapt method
                cloned_preprocessor.adapt(feature_ds)
                self.adapted_preprocessors[feature] = cloned_preprocessor

    def adapt_pandas_dataset(self, dataset):
        # To be implemented
        return []

    def adapt(self, dataset):
        if isinstance(dataset, tf.data.Dataset):
            return self.adapt_tf_dataset(dataset)

    def call(self, inputs):
        forward_pass_list = {}
        for name, _, features in self.feature_preprocessor_list:
            forward_pass_list[name] = [
                self.adapted_preprocessors[feature](inputs[feature])
                for feature in features
            ]
        return forward_pass_list
