"""Base Classes for encoders using tensrflow feature columns"""

from abc import ABC, abstractmethod

import tensorflow as tf


class BaseFeatureColumnEncoder(ABC):
    """Base class for a tensorlow feature column based encoder"""

    def __init__(self, feature_transformer=None, **kwargs):
        self.feature_transformer = feature_transformer
        self.kwargs = kwargs

    def create_inputs(self, dataset=None, features=None):
        """Create model inputs

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict): Keras inputs for each feature
        """
        def get_data_type(ds, feature):
            """helper function to get dtype of feature"""
            if len(ds.element_spec) == 2:
                return ds._structure[0][feature].dtype
            return ds._structure[feature].dtype

        return {feature: tf.keras.Input(shape=(1,), name=feature, dtype=get_data_type(dataset, feature))\
            for feature in features}

    # should perhaps be a abstractmethod
    def encode(self, dataset=None, features=None):
        """Apply feature encoding. This method can be over ridden for specific use cases

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (list): list of encoded features
        """
        return [self.feature_transformer(feature, **self.kwargs) for feature in features]

    def transform(self, dataset=None, features=None):
        """Encoding numerical type features with tensorflow feature columns

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (dict, list): Keras inputs for each feature and list of encoders
        """
        return self.create_inputs(dataset, features), self.encode(dataset, features)


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

    if isinstance(dataset.element_spec, tuple):
        map_func = lambda x, y: x[feature]
    else:
        map_func = lambda x: x[feature]

    feature_vocab_list = {}
    for feature in features:
        feature_ds = dataset.map(map_func)\
                            .apply(tf.data.experimental.unique())\
                            .as_numpy_iterator()
        uniq_vocab = list(feature_ds)
        if all(map(lambda x: isinstance(x, bytes), uniq_vocab)):
            # map bytes to ensure objects are serializable when saving model
            uniq_vocab = [str(value, 'utf-8') for value in uniq_vocab]
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
