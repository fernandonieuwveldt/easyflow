"""Testing for preprocessing pipeline module
"""

import unittest

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# local imports
from easyflow.data import TensorflowDataMapper
from easyflow.preprocessing import FeatureUnion
from easyflow.preprocessing import (
    FeatureInputLayer,
    PreprocessorChain,
)


class TestPreprocessingPipelines(unittest.TestCase):
    """Test the Preprocessing module pipelines
    """

    def setUp(self):
        dataframe = pd.read_csv("easyflow/tests/test_data/heart.csv")
        labels = dataframe.pop("target")
        dataset_mapper = TensorflowDataMapper()
        self.dataset = dataset_mapper.map(dataframe, labels).batch(32)

        self.numerical_features = [
            "age",
            "trestbps",
            "chol",
            "thalach",
            "oldpeak",
            "slope",
        ]
        self.categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "ca"]
        # thal is represented as a string
        self.string_categorical_features = ["thal"]
        self.dtype_mapper = {
            "age": tf.float32,
            "sex": tf.float32,
            "cp": tf.float32,
            "trestbps": tf.float32,
            "chol": tf.float32,
            "fbs": tf.float32,
            "restecg": tf.float32,
            "thalach": tf.float32,
            "exang": tf.float32,
            "oldpeak": tf.float32,
            "slope": tf.float32,
            "ca": tf.float32,
            "thal": tf.string,
        }
        self.all_feature_inputs = FeatureInputLayer(self.dtype_mapper)

        self.feature_preprocessor_list = [
            ("numeric_encoder", layers.Normalization(), self.numerical_features),
            (
                "categorical_encoder",
                layers.IntegerLookup(output_mode="binary"),
                self.categorical_features,
            ),
            # For feature thal we first need to run StringLookup followed by a IntegerLookup layer
            (
                "string_encoder",
                PreprocessorChain(
                    [layers.StringLookup(), layers.IntegerLookup(output_mode="binary")]
                ),
                self.string_categorical_features,
            ),
        ]

    def test_preprocessing_pipeline(self):
        """Test Feature union and model fit
        """
        all_feature_inputs, history = train_model_util(
            self.dataset, self.all_feature_inputs, self.feature_preprocessor_list
        )
        assert len(all_feature_inputs) == 13
        # test if the model ran through all 10 epochs
        assert len(history.history["loss"]) == 10

    def test_identity_layer(self):
        """Test that if preprocessor is None, the IdentityPreprocessing layer should be applied
        """
        feature_encoder_list_none = [
            ("numeric_encoder", None, self.numerical_features),
            (
                "categorical_encoder",
                layers.IntegerLookup(output_mode="binary"),
                self.categorical_features,
            ),
            # For feature thal we first need to run StringLookup followed by a IntegerLookup layer
            (
                "string_encoder",
                PreprocessorChain(
                    [layers.StringLookup(), layers.IntegerLookup(output_mode="binary")]
                ),
                self.string_categorical_features,
            ),
        ]

        all_feature_inputs, history = train_model_util(
            self.dataset, self.all_feature_inputs, feature_encoder_list_none
        )
        assert len(all_feature_inputs) == 13
        # test if the model ran through all 10 epochs
        assert len(history.history["loss"]) == 10

    def test_preprocessing_layer_with_no_args(self):
        """Test preprocessing layers with no arguments supplied
        """
        steps_list = [
            (
                "string_encoder",
                PreprocessorChain(
                    [layers.StringLookup(), layers.IntegerLookup(output_mode="binary")]
                ),
                self.string_categorical_features,
            )
        ]
        preprocessor = FeatureUnion(steps_list)
        preprocessor.adapt(self.dataset)
        preprocessing_layer = preprocessor(self.all_feature_inputs)
        assert preprocessing_layer.shape[-1] == 6

    def test_encoding_validator(self):
        """Test preprocessing layers with arguments supplied
        """
        steps_list = [
            (
                "string_encoder",
                PreprocessorChain(
                    [
                        layers.StringLookup(max_tokens=4),
                        layers.IntegerLookup(output_mode="binary"),
                    ]
                ),
                self.string_categorical_features,
            )
        ]
        preprocessor = FeatureUnion(steps_list)
        preprocessor.adapt(self.dataset)
        preprocessing_layer = preprocessor(self.all_feature_inputs)
        assert preprocessing_layer.shape[-1] == 5

    def test_infered_pipeline(self):
        """test infered pipeline"""
        preprocessor = FeatureUnion.from_infered_pipeline(self.dataset)
        preprocessor.adapt(self.dataset)
        preprocessing_layer = preprocessor(self.all_feature_inputs)
        # setup simple network
        x = tf.keras.layers.Dense(128, activation="relu")(preprocessing_layer)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs=self.all_feature_inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
        history = model.fit(self.dataset, epochs=10)
        assert len(self.all_feature_inputs) == 13
        # test if the model ran through all 10 epochs
        assert len(history.history["loss"]) == 10


def train_model_util(dataset, all_feature_inputs, feature_preprocessor_list):
    """help function for testing

    Args:
        dataset (tf.data.Dataset): Features Data to apply encoder on.
        feature_encoding_list (list): [List of encoders of the form: ('name', encoder type, list of features)
    """
    preprocessor = FeatureUnion(feature_preprocessor_list)
    preprocessor.adapt(dataset)
    preprocessing_layer = preprocessor(all_feature_inputs)

    # setup simple network
    x = tf.keras.layers.Dense(128, activation="relu")(preprocessing_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=all_feature_inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    history = model.fit(dataset, epochs=10)
    return all_feature_inputs, history


if __name__ == "__main__":
    unittest.main()
