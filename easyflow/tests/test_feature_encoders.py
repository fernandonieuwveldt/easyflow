import unittest
import pandas as pd
import tensorflow as tf

# local imports
from easyflow.data.mapper import TensorflowDataMapper
from easyflow.feature_encoders.transformer import FeatureUnionTransformer
from easyflow.feature_encoders.feature_encoder import NumericalFeatureEncoder, CategoricalFeatureEncoder


class TestFeatureEncoders(unittest.TestCase):
    """Test feature encoder module
    """
    def setUp(self):
        """Setup test data and pipeline objects
        """
        dataframe = pd.read_csv('easyflow/tests/test_data/heart.csv')
        labels = dataframe.pop("target")
        dataset_mapper = TensorflowDataMapper()
        self.dataset = dataset_mapper.map(dataframe, labels).batch(32)

        self.numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']
        self.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal']

    def test_feature_encoder_pipelines(self):
        """Test the feature encoder pipeline by applying a full model
        """
        feature_encoder_list = [('numerical_features', NumericalFeatureEncoder(), self.numerical_features),
                                ('categorical_features', CategoricalFeatureEncoder(), self.categorical_features)]

        feature_layer_inputs, feature_encoders =  FeatureUnionTransformer(feature_encoder_list).transform(self.dataset)
        x = tf.keras.layers.DenseFeatures(feature_encoders)(feature_layer_inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])

        history=model.fit(self.dataset, epochs=10)

        # test if the model ran through all 10 epochs
        assert len(history.history['loss']) == 10
        assert len(feature_encoders) == 13
        assert len(feature_layer_inputs) == 13
        assert tf.keras.layers.DenseFeatures(feature_encoders)(feature_layer_inputs).shape[1] == 29


if __name__ == '__main__':
    unittest.main()
