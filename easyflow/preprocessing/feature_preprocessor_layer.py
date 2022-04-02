"""base classes for stateful preprocessing layers"""
from abc import ABC, abstractmethod
import tensorflow as tf

from easyflow.preprocessing.custom import NumericPreprocessingLayer


def one2one_func(x):
    """helper method to apply one to one preprocessor"""
    return x


def extract_feature_column(dataset, name):
    feature = dataset.map(lambda x, y: x[name])
    feature = feature.map(lambda x: tf.expand_dims(x, -1))
    return feature


class FeaturePreprocessor(tf.keras.layers.Layer):
    """Apply column based transformation on the data using tf.keras  preprocessing layers.

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """

    def __init__(self, feature_preprocessor_list=[], *args, **kwargs):
        super(FeaturePreprocessor, self).__init__(*args, *kwargs)
        feature_preprocessor_list = self.map_preprocessor(feature_preprocessor_list)
        self.feature_preprocessor_list = feature_preprocessor_list
        self.adapted_preprocessors = {}

    def map_preprocessor(self, steps):
        """Check and Map input if any of the preprocessors are None, i.e. use as is. For 
        example Binary features that don't need further preprocessing        
        """
        selector = lambda _preprocessor: _preprocessor or NumericPreprocessingLayer()
        return [
            (name, selector(preprocessor), step) for name, preprocessor, step in steps
        ]

    def adapt_tf_dataset(self, dataset):
        return dataset

    def adapt_pandas_dataset(self, dataset):
        return dataset

    def adapt(self, dataset):
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

    def call(self, inputs):
        forward_pass_list = []
        for _, _, features in self.feature_preprocessor_list:
            forward_pass_list.extend(
                [
                    self.adapted_preprocessors[feature](inputs[feature])
                    for feature in features
                ]
            )
        # should we return each step in dict with stepname as key?
        return tf.keras.layers.concatenate(forward_pass_list)
