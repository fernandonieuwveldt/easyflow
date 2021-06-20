"""
Interface to use tensorflow feature columns with Keras. Only most used encoders
"""
import tensorflow as tf

from .base import BaseFeatureColumnEncoder


def get_unique_vocabulary(dataset=None, features=None):
    """Get feature vocabulary list

    Args:
        dataset (tf.data.Dataset): Features Data to apply encoder on.
        features (list): list of feature names

    Returns:
        (dict): dictionary containing list of unique vocabulary values for each feature
    """
    if hasattr(dataset, '_batch_size'):
        # unbatch dataset
        dataset = dataset.unbatch()

    feature_vocab_list = {}
    for feature in features:
        feature_ds = dataset.map(lambda x, y: x[feature])\
                            .apply(tf.data.experimental.unique())\
                            .as_numpy_iterator()
        uniq_vocab = list(feature_ds)
        feature_vocab_list[feature] = tf.feature_column.categorical_column_with_vocabulary_list(feature, uniq_vocab)
    return feature_vocab_list


class BaseCategoricalFeatureColumnEncoder(BaseFeatureColumnEncoder):
    """Base class for categorical type encoders
    """
    def __init__(self, feature_transformer=None, **kwargs):
        super().__init__(feature_transformer=feature_transformer, **kwargs)

    def encode(self, dataset=None, features=None):
        """Apply feature encoding that requires unique vocabulary as input

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (list): list of encoded features
        """
        feature_vocab_list = get_unique_vocabulary(dataset, features)
        return [self.feature_transformer(feature_vocab_list[feature], **self.kwargs) for feature in features]


class CategoricalFeatureEncoder(BaseCategoricalFeatureColumnEncoder):
    """
    Class encodes Categorical features using tensorflow feature_columns
    """
    def __init__(self, **kwargs):
        super().__init__(feature_transformer=tf.feature_column.indicator_column, **kwargs)


class EmbeddingFeatureEncoder(BaseCategoricalFeatureColumnEncoder):
    """
    Class encodes high cardinality Categorical features(Embeddings) using tensorflow feature_columns
    """
    def __init__(self, **kwargs):
        super().__init__(feature_transformer=tf.feature_column.embedding_column, **kwargs)


class CategoryCrossingFeatureEncoder(BaseCategoricalFeatureColumnEncoder):
    """Create cross column features
    """
    def __init__(self, **kwargs):
        super().__init__(feature_transformer=tf.feature_column.crossed_column, **kwargs)

    def encode(self, dataset=None, features=None):
        """Apply Cross column feature engineering follow by indicator column

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (list): list of encoded features
        """
        feature_vocab_list = get_unique_vocabulary(dataset, features)
        crossed_features = self.feature_transformer(feature_vocab_list, **self.kwargs)
        crossed_features = tf.feature_column.indicator_column(crossed_features)
        return crossed_features
