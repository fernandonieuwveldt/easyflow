"""Custom preprocessing layers"""

from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
import tensorflow as tf


class IdentityPreprocessingLayer(PreprocessingLayer):
    """Helper class to apply no preprocessing and use feature as is
    """
    def call(self, inputs):
        return tf.keras.layers.Reshape((1, ))(inputs)

    def get_config(self): 
        """Override get_config to ensure saving of models works"""
        return super().get_config().copy()
