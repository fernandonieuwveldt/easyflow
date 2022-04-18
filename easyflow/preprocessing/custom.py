"""Custom preprocessing layers"""

from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
import tensorflow as tf
from tensorflow.keras import layers


class FeatureInputLayer:
    """Create model inputs for each Feature in the dataset of features that will be used for training. There are two
    options to create the input layer. By default we will be taking in a dict mapper with feature name and dtype as input to
    construct inputs to the network. Second option is to infer the dtype from Dataset's element spec if source data comes from
    tf.data.Dataset type.

    Args:
        data_type_mapper (dict): Dictionary with feature as key and dtype as value
                                For example {'age': tf.float32, ...}
    Returns:
        (dict): Keras Input for each feature
    """

    def __new__(cls, dtype_mapper):
        """Create Input Layer with type mapper by default and return dict of Input objects.

        Args:
            dtype_mapper (dict): Key and Value pairs for feature name as key and data type as value

        Returns:
            dict: Dict with feaure name as key and  Input object value
        """
        return cls.from_dict_mapper(dtype_mapper)

    @classmethod
    def from_dict_mapper(cls, dtype_mapper):
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

    @classmethod
    def infer_from_data(cls, dataset):
        """Create model inputs from the dataset's element spec.

        Args:
            input (tf.data.Dataset): Training dataset with features

        Returns:
            (dict): Keras Input for each feature
        """

        def get_data_spec(ds):
            """helper function to get extract element spec based on Dataset spec"""
            if len(ds.element_spec) == 2:
                return ds.element_spec[0]
            return ds.element_spec

        spec = get_data_spec(dataset)

        return {
            feature: tf.keras.Input(shape=(1,), name=feature, dtype=tspec.dtype)
            for feature, tspec in spec.items()
        }


class NumericPreprocessingLayer(PreprocessingLayer):
    """Helper class to apply no preprocessing and use feature as is"""

    def call(self, inputs):
        return tf.keras.layers.Reshape((1,))(inputs)

    def get_config(self):
        """Override get_config to ensure saving of models works"""
        return super().get_config().copy()

    def update_state(self, data):
        return dict()


class Pipeline(tf.keras.layers.Layer):
    """Preprocessing layer that chains one or more layers in a sequential order by
    subclassinig Layer class.

    Args:
        layers_to_adapt (list): List of layer that needs to be adapted
    """

    def __init__(self, layers_to_adapt, **kwargs):
        super(Pipeline, self).__init__(**kwargs)
        if not isinstance(layers_to_adapt, (list, tuple)):
            layers_to_adapt = [layers_to_adapt]
        self.layers_to_adapt = layers_to_adapt
        self.adapted_layers = []
        self.pipeline = None

    def adapt(self, data):
        """Adapt layers to adapt sequentially and store adapted layers in executed order in adapted layer list.

        Args:
            data (tf.data.Dataset): Mapped tf.data.Dataset containing only the single feature
        """
        for layer in self.layers_to_adapt:
            if hasattr(layer, 'adapt'):
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


class PreprocessorChain(tf.keras.models.Sequential):
    """Preprocessing model that chains one or more layers in a sequential order by subclassing
    Sequential model class. The functionality is the same as Pipeline.
    """

    def __init__(self, layers_to_adapt=[], **kwargs):
        super(PreprocessorChain, self).__init__(layers=[], **kwargs)
        if not isinstance(layers_to_adapt, (list, tuple)):
            layers_to_adapt = [layers_to_adapt]
        self.layers_to_adapt = layers_to_adapt

    def adapt(self, data):
        """Adapt layers to adapt sequentially and add layers with the tf.keras.models.Sequential add 
        method.

        Args:
            data (tf.data.Dataset): Mapped tf.data.Dataset containing only the single feature
        """
        for counter, layer in enumerate(self.layers_to_adapt):
            if hasattr(layer, 'adapt'):
                layer.adapt(data)

            super().add(layer)

            if len(self.layers_to_adapt) > counter:
                data = (
                    data.map(layer)
                    if isinstance(data, tf.data.Dataset)
                    else layer(data)
                )

    def get_config(self):
        """Update config with layers_to_adapt attr

        Returns:
            dict: Updated config
        """
        config = super().get_config()
        config.update({"layers_to_adapt": self.layers_to_adapt})
        return config


class MultiOutputTransformer(tf.keras.layers.Layer):
    """Applies different feature transformations or preprocessing on the same feature and concatenates
    the transformations into a single layer containing output of N transformations.
    """
    # we should be able to use this class the same way we do FeaturePreprocessor class, i.e standalone
    def __init__(self, steps=[], **kwargs):
        super(MultiOutputTransformer, self).__init__(**kwargs)
        if not isinstance(steps, (list, tuple)):
            steps = [steps]
        self.steps = steps
        self.processed_layers = []

    def adapt(self, data):
        """Apply different or independent preprocessing steps on the same data. The results will be concatenated
        into a single layer.

        Args:
            data (tf.data.Dataset): Mapped tf.data.Dataset containing only the single feature
        """
        for layer in self.steps:
            if hasattr(layer, 'adapt'):
                layer.adapt(data)
            self.processed_layers.append(layer)

    def call(self, inputs):
        """Apply sequential model containing adapted layers.

        Args:
            inputs (tf.data.Dataset): Feature to be adapted as a Mapped Dataset

        Returns:
            tf.Tensor: returns output after applying adapted layers.
        """
        return tf.keras.layers.concatenate([
            processed_layer(inputs) for processed_layer in self.processed_layers
        ])

    def get_config(self):
        """Update config with layers_to_adapt attr

        Returns:
            dict: Updated config
        """
        config = super().get_config()
        config.update(
            {"steps": self.steps, "processed_layers": self.processed_layers}
        )
        return config


def StringToIntegerLookup(**kwargs):
    """Implements a common pipeline when categorical features are of type string.
    Steps involve to first apply StringLookup followed by IntegerLookup.

    Args:
        kwargs: All arguments are related to IntegerLookup
    """
    return PreprocessorChain(
            [layers.StringLookup(), layers.IntegerLookup(output_mode='binary', **kwargs)]   
    )
