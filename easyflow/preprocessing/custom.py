"""Custom preprocessing layers"""

from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
import tensorflow as tf


def FeatureInputLayer(dtype_mapper={}):
    """Create model inputs
    Args:
        data_type_mapper (dict): Dictionary with feature as key and dtype as value
                                For example {'age': tf.float32, ...}
    Returns:
        (dict): Keras Input for each feature
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
    """Preprocessing layer that chains one or more layer in sequential order by 
    subclassinig Layer class

    Args:
        layers_to_adapt (list): List of layer that needs to be adapted

    """

    def __init__(self, layers_to_adapt, **kwargs):
        super(PreprocessingChainer, self).__init__(**kwargs)
        if not isinstance(layers_to_adapt, (list, tuple)):
            layers_to_adapt = [layers_to_adapt]
        self.layers_to_adapt = layers_to_adapt
        self.adapted_layers = []
        self.pipeline = None

    def adapt(self, data):
        """Adapt layers to adapt sequentially

        Args:
            data (tf.data.Dataset): Mapped tf.data.Dataset containing only the single feature
        """
        for layer in self.layers_to_adapt:
            layer.adapt(data)
            self.adapted_layers.append(layer)
            if len(self.layers_to_adapt) >= 2:
                data = (
                    data.map(layer)
                    if isinstance(data, tf.data.Dataset)
                    else layer(data)
                )

        self.pipeline = tf.keras.models.Sequential(self.adapted_layers)

    def call(self, inputs):
        """Apply sequential model containing adapted layers.

        Args:
            inputs (tf.data.Dataset): Feature to be adapted as a Mapped Dataset

        Returns:
            tf.Tensor: returns output after applying adapted layers.
        """
        return self.pipeline(inputs)

    def get_config(self):
        """Update config with layers_to_adapt attr

        Returns:
            dict: Updated config
        """
        config = super().get_config()
        config.update(
            {"layers_to_adapt": self.layers_to_adapt, "pipeline": self.pipeline}
        )
        return config


class SequentialPreprocessingChainer(tf.keras.models.Sequential):
    """Preprocessing model that chains one or more layers in sequential order by subclassing 
    Sequential model class.
    """

    def __init__(self, layers_to_adapt=[], **kwargs):
        super(SequentialPreprocessingChainer, self).__init__(layers=[], **kwargs)
        if not isinstance(layers_to_adapt, (list, tuple)):
            layers_to_adapt = [layers_to_adapt]
        self.layers_to_adapt = layers_to_adapt

    def adapt(self, data):
        """Adapt layers to adapt sequentially

        Args:
            data (tf.data.Dataset): Mapped tf.data.Dataset containing only the single feature
        """
        for counter, layer in enumerate(self.layers_to_adapt, start=0):
            layer.adapt(data)
            super().add(layer)
            if len(self.layers_to_adapt) > counter:
                data = data.map(layer)

    def get_config(self):
        """Update config with layers_to_adapt attr

        Returns:
            dict: Updated config
        """
        config = super().get_config()
        config.update(
            {"layers_to_adapt": self.layers_to_adapt}
        )
        return config
