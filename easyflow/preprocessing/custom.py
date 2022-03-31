"""Custom preprocessing layers"""

from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
import tensorflow as tf


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


class Pipeline(tf.keras.layers.Layer):
    def __init__(self, layers_to_adapt, **kwargs):
        super(Pipeline, self).__init__(**kwargs)
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


class _Pipeline(tf.keras.models.Sequential):
    def __init__(self, layers_to_adapt=[], **kwargs):
        super(_Pipeline, self).__init__(layers=[], **kwargs)
        if not isinstance(layers_to_adapt, (list, tuple)):
            layers_to_adapt = [layers_to_adapt]
        self.layers_to_adapt = layers_to_adapt

    def adapt(self, data):
        # perhaps recursion will sort out issue below
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

    # def feature_extractor(self, dataset):
    #     # extract_and_combine = tf.data.Dataset.zip(
    #     #     tuple(extract_feature_column(dataset, feature) for feature in features)
    #     #     ).as_numpy_iterator()
    #     features = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]
    #     return {feature: tf.expand_dims(dataset[feature], -1) for feature in features}

    # def _adapt(self, dataset):
    #     features_data = dataset.map(lambda features, _: features)
    #     for name, preproc_steps, features in self.feature_preprocessor_list:
    #         feature_ds = features_data.map(self.feature_extractor)
    #         # feature_ds = tf.keras.layers.concatenate(feature_ds)
    #         pipeline = Pipeline(layers_to_adapt=preproc_steps)
    #         pipeline.adapt(feature_ds)
    #         self.adapted_preprocessors[features] = pipeline
    # def call(self, inputs):
    #     forward_pass_list = []
    #     for _, _, features in self.feature_preprocessor_list:
    #         for feature in features:
    #             feature_ds = inputs[feature]
    #             # feature_ds = extract_feature_column(inputs, feature)
    #             # layer = self.adapted_preprocessors[feature]
    #             feature_ds = self.adapted_preprocessors[feature](inputs[feature])
    #             # feature_ds = feature_ds.map(self.adapted_preprocessors[feature])
    #             forward_pass_list.append(feature_ds)
    #     return tf.keras.layers.concatenate(forward_pass_list)
