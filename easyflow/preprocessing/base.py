"""base classes for stateful preprocessing layers"""
import tensorflow as tf


def extract_feature_column(dataset, name):
    dataset = dataset.map(lambda x, y: x[name])
    dataset = dataset.map(lambda x: tf.expand_dims(x, -1))
    return dataset


class BaseFeatureEncoder:
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
            input_feature ([str]): feature to encode
            name ([str]): feature to encode
            dataset (tf.Data): [description]

        Returns:
            tf.Data.dataset: encoded feature
        """
        feature_ds = extract_feature_column(dataset, name)
        encoded_feature = self.encoder(input_feature)
        return encoded_feature


class BaseKerasInput:
    """
    Base class for the inputs to the neural network
    """
    def __init__(self, features=None, feature_encoder=None, dtype=None):
        self.features = features
        if not isinstance(self.features, list):
            self.features = [self.features]
        self.feature_encoder = feature_encoder
        self.dtype = dtype

    def create_inputs(self):
        """Create inputs for Keras Model

        Returns:
            list: list of keras inputs
        """
        return [tf.keras.Input(shape=(1,), name=feature, dtype=self.dtype) for feature in self.features]

    def encode_input_features(self, all_inputs=None, dataset=None):
        """ Encode Input with specified preprocessing layer

        Args:
            all_inputs list: list of Keras inputs
            dataset (tf.Data.dataset): data the preprocessing layer

        Returns:
            list: list of encoded features
        """

        return [self.feature_encoder.encode(all_inputs[k], feature, dataset)\
            for (k, feature) in enumerate(self.features)]
