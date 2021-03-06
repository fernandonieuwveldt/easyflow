"""base classes for stateful preprocessing layers"""
import tensorflow as tf


def extract_feature_column(dataset, name):
    dataset = dataset.map(lambda x, y: x[name])
    dataset = dataset.map(lambda x: tf.expand_dims(x, -1))
    return dataset


class BaseFeaturePreprocessor:
    """
    Base class for encoding features with Keras preprocessing layers
    """
    def __init__(self, encoder=None):
        self.encoder = encoder

    def adapt(self, feature_dataset=None):
        """Apply stateful preprocessing layer

        Args:
            feature_dataset (tf.Data.dataset): feature data 

        Returns:
            encoded features
        """
        self.encoder.adapt(feature_dataset)
        return self

    def encode(self, input_feature, name, dataset):
        """Encoded dataset with adapted encoder

        Args:
            input_feature (str): feature to encode
            name (str): feature to encode
            dataset (tf.Data): [description]

        Returns:
            tf.Data.dataset: encoded feature
        """
        feature_ds = extract_feature_column(dataset, name)
        encoded_feature = self.encoder(input_feature)
        return encoded_feature


class BasePreprocessingColummnTransformer:
    """
    Base class for the inputs to the neural network
    """
    def __init__(self, feature_encoder=None):
        self.encoder = BaseFeaturePreprocessor(feature_encoder)

    def create_inputs(self, features):
        """Create inputs for Keras Model

        Returns:
            list: list of keras inputs
        """
        return [tf.keras.Input(shape=(), name=feature, dtype="int64") for feature in features]

    def encode_input_features(self, features=None, all_inputs=None, dataset=None):
        """ Encode Input with specified preprocessing layer

        Args:
            all_inputs (list): list of Keras inputs
            dataset (tf.Data.dataset): data the preprocessing layer

        Returns:
            list: list of encoded features
        """
        return [self.encoder.encode(all_inputs[k], feature, dataset)\
            for (k, feature) in enumerate(features)]

    def encode(self, dataset, features):
        feature_inputs = self.create_inputs(features)
        encoded_features = self.encode_input_features(features, feature_inputs, dataset)
        return feature_inputs, encoded_features


class PreprocessorColumnTransformer:
    """Apply column based transformation on the data

    Args:
        preprocessing_list : 
    """
    def __init__(self, feature_encoder_list=None):
        self.feature_encoder_list = feature_encoder_list
        
    def transform(self, dataset):
        """Apply feature encodings on supplied list

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        feature_layer_inputs = {}
        feature_encoders = []
        for (name, preprocessor, features) in self.feature_encoder_list:
            preprocessor = BasePreprocessingColummnTransformer(preprocessor)
            feature_inputs, feature_encoded = preprocessor.encode(dataset, features)
            feature_layer_inputs[name] = feature_inputs
            feature_encoders.extend(feature_encoded)
        return feature_layer_inputs, feature_encoders


class PreprocessorUnionTransformer(PreprocessorColumnTransformer):
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
    from tensorflow.keras.layers.experimental.preprocessing import Normalization, CategoryEncoding

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import pandas
    # import numpy
    # from classifier import MOAClassifier


    train_data_features = pandas.read_csv("moa/data/lish-moa/train_features.csv").drop('sig_id', axis=1)
    train_data_features['cp_time'] = train_data_features['cp_time'].map(str)
    raw_labels = pandas.read_csv("moa/data/lish-moa/train_targets_scored.csv").drop('sig_id', axis=1)

    data_types = train_data_features.dtypes
    categorical_features = ['cp_type', 'cp_dose', 'cp_time']
    numerical_features = data_types[data_types=='float64'].index.tolist()
    numerical_features_gene = [feature for feature in numerical_features if 'g' in feature]
    numerical_features_cell = [feature for feature in numerical_features if 'c' in feature]

    from easyflow.data import TFDataTransformer
    dataset = TFDataTransformer().transform(train_data_features,
                                            raw_labels).batch(512)
    feature_encoder_list = [
                            ('numeric_gene_encoder', Normalization(), numerical_features_gene[:5]),
                            #('numeric_cell_encoder', Normalization(), numerical_features_cell[:5]),
                            # ('categorical_encoder', CategoryEncoding(max_tokens=3, output_mode="binary"), categorical_features)
                            ]
    preprocessor = PreprocessorUnionTransformer(feature_encoder_list)
    feature_layer_inputs, feature_encoders = preprocessor.transform(dataset)

    features = tf.keras.layers.Dense(10)(feature_encoders)
    outputs = tf.keras.layers.Dense(206, activation='sigmoid')(features)
    model = tf.keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
    )
    model.fit(dataset)
