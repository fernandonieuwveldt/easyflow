"""Custom preprocessing layers"""

from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
import tensorflow as tf


class IdentityLayer(PreprocessingLayer):
    """Helper class to apply on features where no preprocessing as needed and use as is
    """
    def call(self, inputs):
        return tf.keras.layers.Reshape((1, ))(inputs)
