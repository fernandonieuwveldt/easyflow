"""
Interface to use tensorflow feature columns with Keras for Numerical features
"""

import tensorflow as tf

from .base import BaseFeatureColumnEncoder


class NumericalFeatureEncoder(BaseFeatureColumnEncoder):
    """
    Class encodes numerical features using tensorflow feature_columns. This is a wrapper to
    tf.feature_column.numerical_column to conform to the BaseFeatureColumnEncoder interface and does not change the behaviour.

    Examples
    --------
    >>> data = {'feature': [1.1, 1.2, 0.0, 2.2]}
    >>> dataset = tf.data.Dataset.from_tensor_slices(data).batch(4)
    >>> example_batch = next(iter(dataset))
    >>> feature_layer = lambda feature_column, batch: tf.keras.layers.DenseFeatures(feature_column)(batch).numpy()
    >>> encoder = NumericalFeatureEncoder()
    >>> encoded_feature = encoder.encode(dataset, ['feature'])
    >>> feature_layer(encoded_feature, example_batch)
        array([[1.1],
               [1.2],
               [0. ],
               [2.2]], dtype=float32)
    """
    def __init__(self, **kwargs):
        super().__init__(feature_transformer=tf.feature_column.numeric_column, **kwargs)


class BucketizedFeatureEncoder(BaseFeatureColumnEncoder):
    """Class applies bucketized encoding by first applying NumericalFeatureEncoder. This is a wrapper to
    tf.feature_column.bucketized_column to conform to the BaseFeatureColumnEncoder interface and does not change the behaviour.

    Examples
    --------
    >>> data = {'feature': [1.1, 1.2, 0.0, 2.2]}
    >>> dataset = tf.data.Dataset.from_tensor_slices(data).batch(4)
    >>> example_batch = next(iter(dataset))
    >>> feature_layer = lambda feature_column, batch: tf.keras.layers.DenseFeatures(feature_column)(batch).numpy()
    >>> encoder = BucketizedFeatureEncoder(boundaries=[1])
    >>> encoded_feature = encoder.encode(dataset, ['feature'])
    >>> feature_layer(encoded_feature, example_batch)
        array([[0., 1.],
               [0., 1.],
               [1., 0.],
               [0., 1.]], dtype=float32)
    """
    def __init__(self, **kwargs):
        super().__init__(feature_transformer=tf.feature_column.bucketized_column, **kwargs)

    def encode(self, dataset=None, features=None):
        """Apply bucketized feature encoding by first applying numerical encoder.

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.
            features (list): list of feature names

        Returns:
            (list): list of encoded features
        """
        numerical_encoder = NumericalFeatureEncoder()
        numerical_inputs, numerical_encoded_features = numerical_encoder.transform(dataset, features)
        return [self.feature_transformer(feature, **self.kwargs) for feature in numerical_encoded_features]
