from abc import ABC, abstractclassmethod
import tensorflow as tf


EARLY_STOPPING_PARAMS = {
    'monitor': 'val_loss',
    'mode': 'min',
    'verbose': 1,
    'patience': 10
}


MODEL_CHECKPOINT_PARAMS = {
    'filepath': 'classifier.h5',
    'monitor': 'val_loss',
    'mode': 'min',
    'verbose': 1,
    'save_best_only': True
}


class BaseClassifier(ABC):
    """Base class for a classifier based on a Keras model

    Args:
        ABC ([type]): [description]
    """
    def __init__(self, train_split_fraction=0.75, batch_size=32, epochs=100):
        self.train_split_fraction = train_split_fraction
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.early_stopping = None
        self.model_checkpoint = None

    @abstractclassmethod
    def compile_model(self, samples):
        """
        set up model and return compiled model
        """

    def split_data_set(self, features=None, target=None):
        """
        Split data set
        """
        rows = features.shape[0]
        training_size = int(self.train_split_fraction * rows)
        dataset = tf.data.Dataset.from_tensor_slices((features, target)).shuffle(rows)
        train_data_set = dataset.take(training_size).batch(self.batch_size)
        val_data_set = dataset.skip(training_size).batch(self.batch_size)
        return train_data_set, val_data_set

    def fit(self, features=None, target=None):
        """
        Fit classifier

        Args:
            features ([type], optional): [description]. Defaults to None.
            target ([type], optional): [description]. Defaults to None.
        """
        train_data_set, val_data_set = self.split_data_set(features, target)
        self.model = self.compile_model(train_data_set)  # check if this correct
        self.early_stopping = tf.keras.callbacks.EarlyStopping(**EARLY_STOPPING_PARAMS)
        self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(**MODEL_CHECKPOINT_PARAMS)
        self.model.fit(train_data_set,
                       validation_data=val_data_set,
                       epochs=self.epochs,
                       callbacks=[self.early_stopping, self.model_checkpoint])
        return self

    def predict_proba(self, x=None):
        saved_model = tf.keras.models.load_model(MODEL_CHECKPOINT_PARAMS['filepath'])
        return saved_model.predict(x)
