"""
Interface to use tensorflow feature columns with Keras. Only most used encoders
"""
import tensorflow as tf

from .base import BaseFeatureColumnEncoder


def get_unique_vocab(dataset=None, features=None):
    """Get feature vocab and create inputs

    Args:
        dataset (tf.data.Dataset): Features Data to apply encoder on.
        features (list): list of feature names

    Returns:
        (dict, list): Keras inputs for each feature and list of encoders
    """
    if hasattr(dataset, '_batch_size'):
        # unbatch dataset
        dataset = dataset.unbatch()

    feature_vocab_list, categorical_inputs = {}, {}
    for feature in features:
        dtype = dataset._structure[0][feature].dtype
        categorical_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=dtype)
        feature_ds = dataset.map(lambda x, y: x[feature])\
                            .apply(tf.data.experimental.unique())\
                            .as_numpy_iterator()
        uniq_vocab = list(feature_ds)
        feature_vocab_list[feature] = tf.feature_column.categorical_column_with_vocabulary_list(feature, uniq_vocab)
    return categorical_inputs, feature_vocab_list


class CategoricalFeatureEncoder(BaseFeatureColumnEncoder):
    """
    Class encodes Categorical features using tensorflow feature_columns
    """
    def __init__(self):
        pass

    def encode(self, dataset=None, features=None):
        """Encoding features as one hot encoded with tensorflow feature columns

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        categorical_inputs, feature_vocab_list = get_unique_vocab(dataset, features)
        feature_encoders = {feature: tf.feature_column.indicator_column(feature_vocab_list[feature]) for feature in features}
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

    def encode(self, dataset=None, features=None):
        """Encoding features as Embeddings with tensorflow feature columns

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        embedding_inputs, feature_vocab_list = get_unique_vocab(dataset, features)
        dimension_fn = lambda vocab: min(int(len(vocab)**self.embedding_space_factor), self.max_dimension)
        feature_encoders = {feature:tf.feature_column.embedding_column(feature_vocab_list[feature],
                                                                       initializer=self.initializer,
                                                                       dimension=dimension_fn(feature_vocab_list[feature]))\
                            for feature in features}
        return embedding_inputs, [feature for _, feature in feature_encoders.items()]


class NumericalFeatureEncoder(BaseFeatureColumnEncoder):
    """
    Class encodes numerical features using tensorflow feature_columns
    """
    def __init__(self):
        pass

    def encode(self, dataset=None, features=None):
        """Encoding numerical type features with tensorflow feature columns

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        numerical_inputs, feature_encoders = {}, {}

        for feature in features:
            numerical_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
            feature_encoders[feature] = tf.feature_column.numeric_column(feature)
        return numerical_inputs, [feature for _, feature in feature_encoders.items()]
