"""Testing for preprocessing pipeline module
"""

import unittest

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization, CategoryEncoding, StringLookup

# local imports
from easyflow.data.mapper import TensorflowDataMapper
from easyflow.preprocessing.preprocessor import Encoder, SequentialEncoder, FeatureUnion


class TestPreprocessingPipelines(unittest.TestCase):
    """Test the Preprocessing module pipelines
    """
    def setUp(self):
        dataframe = pd.read_csv('easyflow/tests/test_data/heart.csv')
        labels = dataframe.pop("target")
        dataset_mapper = TensorflowDataMapper()
        self.dataset = dataset_mapper.map(dataframe, labels).batch(32)

        self.numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']
        self.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca']
        # thal is represented as a string
        self.string_categorical_features = ['thal']

        self.feature_encoder_list = [
                            Encoder([('numeric_encoder', Normalization(), self.numerical_features )]),
                            Encoder([('categorical_encoder', CategoryEncoding(), self.categorical_features)]),
                            # For feature thal we first need to run StringLookup followed by a CategoryEncoding layer
                            SequentialEncoder([('string_encoder', StringLookup(), self.string_categorical_features),
                                               ('categorical_encoder', CategoryEncoding(), self.string_categorical_features)])
                            ]

    def test_preprocessing_pipeline(self):
        """Test Feature union and model fit
        """
        all_feature_inputs, history = train_model_util(self.dataset, self.feature_encoder_list)
        assert len(all_feature_inputs) == 13
        # test if the model ran through all 10 epochs
        assert len(history.history['loss']) == 10

    def test_identity_layer(self):
        """Test that if preprocessor is None, the IdentityPreprocessing layer should be applied
        """
        feature_encoder_list_none = [
                            Encoder([('numeric_encoder', None, self.numerical_features )]),
                            Encoder([('categorical_encoder', CategoryEncoding(), self.categorical_features)]),
                            # For feature thal we first need to run StringLookup followed by a CategoryEncoding layer
                            SequentialEncoder([('string_encoder', StringLookup(), self.string_categorical_features),
                                               ('categorical_encoder', CategoryEncoding(), self.string_categorical_features)])
                            ]

        all_feature_inputs, history = train_model_util(self.dataset, feature_encoder_list_none)
        assert len(all_feature_inputs) == 13
        # test if the model ran through all 10 epochs
        assert len(history.history['loss']) == 10

    def test_preprocessing_layer_with_no_args(self):
        """Test preprocessing layers with no arguments supplied
        """
        steps_list = [
                      SequentialEncoder([('string_encoder', StringLookup(), self.string_categorical_features),
                                         ('categorical_encoder', CategoryEncoding(), self.string_categorical_features)])
        ]
        encoder = FeatureUnion(steps_list)
        all_feature_inputs, preprocessing_layer = encoder.encode(self.dataset)
        assert preprocessing_layer.shape[-1] == 7

    def test_preprocessing_layer_with_args(self):
        """Test preprocessing layers with arguments supplied
        """
        steps_list = [
                      SequentialEncoder([('string_encoder', StringLookup(max_tokens=4), self.string_categorical_features),
                                         ('categorical_encoder', CategoryEncoding(), self.string_categorical_features)])
        ]
        encoder = FeatureUnion(steps_list)
        all_feature_inputs, preprocessing_layer = encoder.encode(self.dataset)
        assert preprocessing_layer.shape[-1] == 4


def train_model_util(dataset=None, feature_encoding_list=None):
    """help function for testing

    Args:
        dataset (tf.data.Dataset): Features Data to apply encoder on.
        feature_encoding_list (list): [List of encoders of the form: ('name', encoder type, list of features)
    """

    encoder = FeatureUnion(feature_encoding_list)
    all_feature_inputs, preprocessing_layer = encoder.encode(dataset)
    # setup simple network
    x = tf.keras.layers.Dense(128, activation="relu")(preprocessing_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=all_feature_inputs, outputs=outputs)
    model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
    )
    history=model.fit(dataset, epochs=10)
    return all_feature_inputs, history


if __name__ == '__main__':
    unittest.main()
