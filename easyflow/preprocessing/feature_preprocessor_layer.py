"""base classes for stateful preprocessing layers"""
import tensorflow as tf
from tensorflow.keras import layers

from easyflow.preprocessing.custom import (
    NumericPreprocessingLayer,
    SequentialPreprocessingChainer,
)


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


# Will be our factory
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
            ("numerical_features", layers.Normalization(), numeric_features),
            (
                "categorical_features",
                layers.IntegerLookup(output_mode="binary"),
                categoric_features,
            ),
            (
                "string_categorical_features",
                SequentialPreprocessingChainer(
                    [layers.StringLookup(), layers.IntegerLookup(output_mode="binary")]
                ),
                string_categoric_features,
            ),
        ]
        return cls(feature_preprocessor_list)

    def map_preprocessor(self, steps):
        """Check and Map input if any of the preprocessors are None, i.e. use as is. For 
        example Binary features that don't need further preprocessing   

        Args:
            steps (list): Preprocessing steps.

        Returns:
            list: List of mapped preprocessors if None was supplied.
        """
        selector = lambda _preprocessor: _preprocessor or NumericPreprocessingLayer()
        return [
            (name, selector(preprocessor), step) for name, preprocessor, step in steps
        ]

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

    def __getitem__(self, idx):
        # This should rather return the adapted layers for the specific step
        return self.feature_preprocessor_list[idx]

    def __len__(self):
        """Total number of steps

        Returns:
            int: Total number of steps
        """
        return len(self.feature_preprocessor_list)

    @property
    def preprocessor_name(self):
        """Return the step names

        Returns:
            list: List of step names
        """
        return [self[k][0] for k in range(len(self))]
