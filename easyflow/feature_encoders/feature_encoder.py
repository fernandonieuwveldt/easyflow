"""
classes for encoding features using tensorflow features columns
"""
import tensorflow as tf


class CategoricalFeatureEncoder:
    """
    Class encodes Categorical features using tensorflow feature_columns
    """
    def __init__(self, features=None):
        self.features = features

    def encode(self, X=None):
        """
        Set inputs and catergorical vocab list
        """
        feature_vocab_list, categorical_inputs, feature_encoders = {}, {}, {}

        for feature in self.features:
            categorical_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.string)
            feature_vocab_list[feature] = tf.feature_column.categorical_column_with_vocabulary_list(feature, X[feature].unique().tolist())
            feature_encoders[feature] = tf.feature_column.indicator_column(feature_vocab_list[feature])
        return categorical_inputs, [feature for _, feature in feature_encoders.items()]


class EmbeddingFeatureEncoder:
    """
    Class encodes high cardinality Categorical features(Embeddings) using tensorflow feature_columns
    """

    def __init__(self, features, embedding_space_factor=0.5):
        self.features = features
        self.embedding_space_factor = embedding_space_factor

    def encode(self, X=None):
        """
        Set inputs and dimension for embedding output space
        """
        feature_vocab_list, embedding_inputs, feature_encoders = {}, {}, {}

        for feature in self.features:
            uniq_vocab = X[feature].unique().tolist()
            embedding_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.string)
            feature_vocab_list[feature] = tf.feature_column.categorical_column_with_vocabulary_list(feature, uniq_vocab)
            feature_encoders[feature] = tf.feature_column.embedding_column(feature_vocab_list[feature],
                                                                           initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
                                                                           dimension=min(int(len(uniq_vocab)**self.embedding_space_factor), 50))
        return embedding_inputs, [feature for _, feature in feature_encoders.items()]


class NumericalFeatureEncoder:
    """
    Class encodes numerical features using tensorflow feature_columns
    """

    def __init__(self, features):
        self.features = features

    def encode(self, X=None):
        """
        Set inputs for numerical features
        """
        numerical_inputs, feature_encoders = {}, {}

        for feature in self.features:
            numerical_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
            feature_encoders[feature] = tf.feature_column.numeric_column(feature)
        return numerical_inputs, [feature for _, feature in feature_encoders.items()]