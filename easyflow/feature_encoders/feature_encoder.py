"""
classes for encoding features using tensorflow features columns
"""
import tensorflow as tf

from .base import BaseFeatureColumnEncoder


class CategoricalFeatureEncoder(BaseFeatureColumnEncoder):
    """
    Class encodes Categorical features using tensorflow feature_columns
    """
    def __init__(self):
        pass

    def encode(self, X=None, features=None):
        """Encoding features as one hot encoded with tensorflow feature columns

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        feature_vocab_list, categorical_inputs, feature_encoders = {}, {}, {}

        for feature in features:
            categorical_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.string)
            feature_vocab_list[feature] = tf.feature_column.categorical_column_with_vocabulary_list(feature, X[feature].unique().tolist())
            feature_encoders[feature] = tf.feature_column.indicator_column(feature_vocab_list[feature])
        return categorical_inputs, [feature for _, feature in feature_encoders.items()]


class EmbeddingFeatureEncoder(BaseFeatureColumnEncoder):
    """
    Class encodes high cardinality Categorical features(Embeddings) using tensorflow feature_columns
    """

    def __init__(self, initializer=None, embedding_space_factor=0.5, max_dimension=50):
        self.initializer = initializer
        if not self.initializer:
            self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.embedding_space_factor = embedding_space_factor
        self.max_dimension = max_dimension

    def encode(self, X=None, features=None):
        """Encoding features as Embeddings with tensorflow feature columns

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        feature_vocab_list, embedding_inputs, feature_encoders = {}, {}, {}

        for feature in features:
            uniq_vocab = X[feature].unique().tolist()
            embedding_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.string)
            feature_vocab_list[feature] = tf.feature_column.categorical_column_with_vocabulary_list(feature, uniq_vocab)
            feature_encoders[feature] = tf.feature_column.embedding_column(feature_vocab_list[feature],
                                                                           initializer=self.initializer,
                                                                           dimension=min(int(len(uniq_vocab)**self.embedding_space_factor), self.max_dimension))
        return embedding_inputs, [feature for _, feature in feature_encoders.items()]


class NumericalFeatureEncoder(BaseFeatureColumnEncoder):
    """
    Class encodes numerical features using tensorflow feature_columns
    """
    def __init__(self):
        pass

    def encode(self, X=None, features=None):
        """Encoding numerical type features with tensorflow feature columns

        Args:
            X (pandas.DataFrame): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        numerical_inputs, feature_encoders = {}, {}

        for feature in features:
            numerical_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
            feature_encoders[feature] = tf.feature_column.numeric_column(feature)
        return numerical_inputs, [feature for _, feature in feature_encoders.items()]
