"""Custom preprocessing layers"""

from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
import tensorflow as tf


def FeatureInputLayer(dtype_mapper={}):
    """Create model inputs
    Args:
        data_type_mapper (dict): Dictionary with feature as key and dtype as value
                                For example {'age': tf.float32, ...}
    Returns:
        (dict): Keras inputs for each feature
    """
    return {
        feature: tf.keras.Input(shape=(1,), name=feature, dtype=dtype)
        for feature, dtype in dtype_mapper.items()
    }


class NumericPreprocessingLayer(PreprocessingLayer):
    """Helper class to apply no preprocessing and use feature as is
    """

    def call(self, inputs):
        return tf.keras.layers.Reshape((1,))(inputs)

    def get_config(self):
        """Override get_config to ensure saving of models works"""
        return super().get_config().copy()

    def update_state(self, data):
        return {}


class PreprocessingChainer(tf.keras.layers.Layer):
    """Preprocessing layer that chains one or more layer in sequential order by subclassinig Layer class
    """
    def __init__(self, layers_to_adapt, **kwargs):
        super(PreprocessingChainer, self).__init__(**kwargs)
        if not isinstance(layers_to_adapt, (list, tuple)):
            layers_to_adapt = [layers_to_adapt]
        self.layers_to_adapt = layers_to_adapt
        self.adapted_layers = []
        # self.pipeline = tf.keras.models.Sequential([])

    def adapt(self, data):
        for layer in self.layers_to_adapt:
            layer.adapt(data)
            self.adapted_layers.append(layer)
            if len(self.layers_to_adapt) >= 2:
                data = data.map(layer)

        self.pipeline = tf.keras.models.Sequential(self.adapted_layers)

    def call(self, inputs):
        return self.pipeline(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"layers_to_adapt": self.layers_to_adapt,}
        )
        return config


class SequentialPreprocessingChainer(tf.keras.models.Sequential):
    """Preprocessing model that chains one or more layers in sequential order by subclassing 
    sequential model class.
    """
    def __init__(self, layers_to_adapt=[], **kwargs):
        super(SequentialPreprocessingChainer, self).__init__(layers=[], **kwargs)
        if not isinstance(layers_to_adapt, (list, tuple)):
            layers_to_adapt = [layers_to_adapt]
        self.layers_to_adapt = layers_to_adapt

    def adapt(self, data):
        for counter, layer in enumerate(self.layers_to_adapt, start=0):
            layer.adapt(data)
            super().add(layer)
            if len(self.layers_to_adapt) > counter:
                data = data.map(layer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"layers_to_adapt": self.layers_to_adapt,}
        )
        return config

