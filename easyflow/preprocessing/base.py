"""base classes for stateful preprocessing layers"""
import tensorflow as tf


def extract_feature_column(dataset, name):
    dataset = dataset.map(lambda x, y: x[name])
    dataset = dataset.map(lambda x: tf.expand_dims(x, -1))
    return dataset


class BaseFeatureTransformer:
    def __init__(self, feature_encoder_list=None):
        self.feature_encoder_list = feature_encoder_list

    def create_inputs(self, features, dtype):
        """Create inputs for Keras Model

        Returns:
            list: list of keras inputs
        """
        # dict would work better
        return [tf.keras.Input(shape=(), name=feature, dtype=dtype) for feature in features]


class FeatureTransformer(BaseFeatureTransformer):
    """Apply column based transformation on the data

    Args:
        preprocessing_list : 
    """
    def transform(self, dataset):
        """Apply feature encodings on supplied list

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        feature_layer_inputs = {}
        feature_encoders = []
        for (name, preprocessor, features, dtype) in self.feature_encoder_list:
            encoded_features = []
            feature_inputs = self.create_inputs(features, dtype)
            for k, feature_name in enumerate(features):
                feature_ds = extract_feature_column(dataset, feature_name)
                preprocessor.adapt(feature_ds)
                encoded_feature = preprocessor(feature_inputs[k])
                encoded_features.append(encoded_feature)
            feature_layer_inputs[name] = feature_inputs
            feature_encoders.extend(encoded_features)
        return feature_layer_inputs, feature_encoders


class PipelineFeatureTransformer(BaseFeatureTransformer):
    """
    Preprocessing pipeline to apply multiple encoders in serie
    """
    def _warm_up(self, name, preprocessor, features, feature_inputs, dtype):
        """Apply feature encodings on supplied list

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        adapted_preprocessors = {}
        encoded_features = {}
        for feature_input, feature_name in zip(feature_inputs, features):
            _preprocessor = preprocessor()
            feature_ds = extract_feature_column(dataset, feature_name)
            _preprocessor.adapt(feature_ds)
            encoded_feature = _preprocessor(feature_input)
            encoded_features[feature_name] = encoded_feature
            adapted_preprocessors[feature_name] = _preprocessor
        return adapted_preprocessors, encoded_features

    def transform(self, dataset):
        name, preprocessor, features, dtype = self.feature_encoder_list[0]
        feature_inputs = self.create_inputs(features, dtype)
        adapted_preprocessors, encoded_features = self._warm_up(name, preprocessor, features, feature_inputs, dtype)
        feature_layer_inputs = {}
        feature_encoders = []
        for (name, preprocessor, features, dtype) in self.feature_encoder_list[1:]:
            for feature_name in features:
                _preprocessor = preprocessor()
                feature_ds = extract_feature_column(dataset, feature_name)
                feature_ds = feature_ds.map(adapted_preprocessors[feature_name])
                _preprocessor.adapt(feature_ds)
                encoded_feature = _preprocessor(encoded_features[feature_name])
                adapted_preprocessors[feature_name] = _preprocessor
                encoded_features[feature_name] = encoded_feature
        return feature_inputs, encoded_features


class FeatureUnionTransformer(FeatureTransformer):
    """Apply column based preprocessing on the data

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, X):
        """Join features. If more flexibility and customization is needed use PreprocessorColumnTransformer.

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.

        Returns:
            (dict, tf.keras.layer): Keras inputs for each feature and concatenated layer
        """
        feature_layer_inputs, feature_encoders = super().transform(X)
        # flatten (or taking the union) of feature encoders 
        return feature_layer_inputs, tf.keras.layers.concatenate(feature_encoders)


if __name__ == '__main__':
    from tensorflow.keras.layers.experimental.preprocessing import Normalization, CategoryEncoding, StringLookup

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import pandas
    # import numpy
    # from classifier import MOAClassifier


    train_data_features = pandas.read_csv("moa/data/lish-moa/train_features.csv").drop('sig_id', axis=1)
    train_data_features['cp_time'] = train_data_features['cp_time'].map(str)
    raw_labels = pandas.read_csv("moa/data/lish-moa/train_targets_scored.csv").drop('sig_id', axis=1)

    data_types = train_data_features.dtypes
    categorical_features = ['cp_type',
                            'cp_dose', 
                            #'cp_time'
                            ]
    numerical_features = data_types[data_types=='float64'].index.tolist()
    numerical_features_gene = [feature for feature in numerical_features if 'g' in feature]
    numerical_features_cell = [feature for feature in numerical_features if 'c' in feature]

    from easyflow.data import TFDataTransformer
    dataset = TFDataTransformer().transform(train_data_features,
                                            raw_labels).batch(512)


    # pipeline_steps = [
    #                   ('indexer', StringIndexer()),
    #                   ('category_encoder', CategoryEncoding())
    # ]
    # categorical_pipeline = Pipeline()

    feature_encoder_list = [
                            # ('numeric_gene_encoder', Normalization(), numerical_features_gene[:5]),
                            # ('numeric_cell_encoder', Normalization(), numerical_features_cell[:2], "float32"),
                            ('string_encoder', StringLookup, categorical_features, "string"),
                            ('categorical_encoder', CategoryEncoding, categorical_features, "float32"),
                            ]
    # preprocessor = PreprocessorUnionTransformer(feature_encoder_list)
    pipeline = PipelineFeatureTransformer(feature_encoder_list)
    feature_layer_inputs, feature_encoders = pipeline.transform(dataset)
    feature_encoders = [v for v in feature_encoders.values()]
    feature_encoders = tf.keras.layers.concatenate(feature_encoders)
    features = tf.keras.layers.Dense(10)(feature_encoders)
    outputs = tf.keras.layers.Dense(206, activation='sigmoid')(features)
    model = tf.keras.Model(inputs=feature_layer_inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
    )
    model.fit(dataset)
