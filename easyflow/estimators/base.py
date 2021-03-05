from abc import ABC, abstractmethod

import tensorflow as tf

from easyflow.data import TFDataTransformer


class BaseClassifier(ABC):
    """Base class for a classifier based on a Keras model
    """
    def __init__(self, train_split_fraction=0.75):
        self.train_split_fraction = train_split_fraction # should not be instance variable

    @abstractmethod
    def compile_model(self, samples, target):
        """Set up network architecture and return compiled model. This method needs to be implemented in the parent class

        Args:
            samples (pandas.DataFrame): Features Data
        """

    def split_data_set(self, features=None, target=None, batch_size=None):
        """Split data for training and validation

        Args:
            features (pandas.DataFrame): Features Data
            target (pandas.Series): Target Data
        
        Returns:
            (tf.data.Dataset, tf.data.Dataset): train and validation datasets
        """
        rows = features.shape[0]
        training_size = int(self.train_split_fraction * rows)
        dataset = TFDataTransformer().transform(features, target)
        train_data_set = dataset.take(training_size).batch(batch_size)
        val_data_set = dataset.skip(training_size).batch(batch_size)
        return train_data_set, val_data_set

    def fit(self, features=None, target=None, batch_size=32, **kwargs):
        """Fit model

        Args:
            features (pandas.DataFrame): Features Data
            target (pandas.Series): Target Data
        """
        self.batch_size = batch_size
        self.model = self.compile_model(features, target)
        train_data_set, val_data_set = self.split_data_set(features, target, batch_size)
        self.model.fit(train_data_set,
                       validation_data=val_data_set,
                       **kwargs)
        return self

    def predict(self, X=None):
        """Apply trained model on data

        Args:
            X (pandas.DataFrame): Features Data

        Returns:
            (numpy.array): array of confidences
        """
        X = TFDataTransformer().transform(X).batch(self.batch_size)
        return self.model.predict(X)
