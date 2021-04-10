"""Testing for preprocessing pipeline module
"""

import unittest

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization, CategoryEncoding, StringLookup

# local imports
from easyflow.data.mapper import TensorflowDataMapper
from easyflow.preprocessing.preprocessor import Encoder, Pipeline, SequentialEncoder, FeatureUnion
from easyflow.preprocessing.custom import IdentityPreprocessingLayer


class TestPreprocessingPipelines(unittest.TestCase):
    """Test the Preprocessing module pipelines
    """
    def setUp(self):
        file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
        dataframe = pd.read_csv(file_url)
        dataframe = dataframe.copy()
        labels = dataframe.pop("target")
        batch_size = 32
        dataset_mapper = TensorflowDataMapper()
        self.dataset = dataset_mapper.map(dataframe, labels)
        train_data_set, val_data_set = dataset_mapper.split_data_set(self.dataset)
        self.train_data_set = train_data_set.batch(batch_size)
        self.val_data_set = val_data_set.batch(batch_size)

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
        encoder = FeatureUnion(self.feature_encoder_list)
        all_feature_inputs, preprocessing_layer = encoder.encode(self.train_data_set)
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

        history=model.fit(self.train_data_set, validation_data=self.val_data_set, epochs=10)

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


if __name__ == '__main__':
    unittest.main()
