"""base classes for stateful preprocessing layers"""
from abc import ABC, abstractmethod
import tensorflow as tf

from .custom import IdentityPreprocessingLayer


def one2one_func(x):
    """helper method to apply one to one preprocessor"""
    return x


def extract_feature_column(dataset, name):
    feature = dataset.map(lambda x, y: x[name])
    feature = feature.map(lambda x: tf.expand_dims(x, -1))
    return feature


class BaseEncoder(ABC):
    """Apply column based transformation on the data using tf.keras  preprocessing layers.

    Args:
        feature_encoder_list : List of encoders of the form: ('name', encoder type, list of features)
    """
    def __init__(self, feature_encoder_list=None):
        self.feature_encoder_list = feature_encoder_list
        _, _, features = self.feature_encoder_list
        # map and validate encoding list
        self.map_preprocessor()
        self.validate_encoding_list()
        self.feature_encoder_list = self.remap(self.feature_encoder_list)
        self.adapted_preprocessors = {feature_name: one2one_func for feature_name in features}

    def validate_encoding_list(self):
        """Validate that all prepocessorts has adapt method"""
        name, preprocessors, features = self.feature_encoder_list
        if not isinstance(preprocessors, list):
            preprocessors = [preprocessors]

        for preprocessor in preprocessors:
            if not hasattr(preprocessor, "adapt"):
                raise TypeError("All preprocessing/encoding layers should have adapt method"
                                "'%s' (type %s) doesn't" % (preprocessor, type(preprocessor)))

    def map_preprocessor(self):
        """Check and Map input if any of the preprocessors are None, i.e. use as is"""
        self.feature_encoder_list = list(self.feature_encoder_list)
        name, preprocessor, features = self.feature_encoder_list
        selector = lambda _preprocessor: _preprocessor or IdentityPreprocessingLayer()
        if isinstance(preprocessor, list):
            self.feature_encoder_list[1] = [selector(_preprocessor) for _preprocessor in preprocessor]
        else:
            self.feature_encoder_list[1] = selector(preprocessor)
        self.feature_encoder_list = tuple(self.feature_encoder_list)

    def remap(self, steps):
        """Map multiple encoders to single encoders. If sequence of encoders are applied on features they will remapped.

        Args:
            steps (list): List of encoders of the form: ('name', encoder type, list of features)

        Returns:
            (list): List of remapped sequential encoders
        """
        name, preprocessors, features = self.feature_encoder_list
        if not isinstance(preprocessors, list):
            return self.feature_encoder_list
        return [(name, preprocessor, features) for preprocessor in preprocessors]

    @abstractmethod
    def encode(self, dataset):
        """Apply feature encodings on supplied feature encoding list

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, dict): Keras inputs for each feature and dict of encoders
        """

    def create_inputs(self, features, dtype):
        """Create list of keras Input layers

        Args:
            features (list): list of feature names
            dtype (str): data type of features

        Returns:
            list: list of Keras Input layers
        """
        return [tf.keras.Input(shape=(1,), name=feature, dtype=dtype) for feature in features]

    def _encode_one(self, dataset, preprocessor, features, feature_inputs):
        """Apply feature encodings on supplied list

        Args:
            dataset (tf.data.Dataset): dataset (tf.data.Dataset): Features Data to apply encoder on.
            preprocessor (tf.keras.layer): tf.keras preprocessing layer
            features (list): list of feature names
            feature_inputs (list): list of Keras Input layers

        Returns:
            (dict): dictionary with feature as key and encoded preprocessing layer as value 
        """
        encoded_features = {}
        # get initial preprocessing layer config 
        config = preprocessor.get_config()
        for k, (feature_input, feature_name) in enumerate(zip(feature_inputs, features)):
            config.pop('name', None)
            _preprocessor = preprocessor if k==0 else preprocessor.from_config(config)
            feature_ds = extract_feature_column(dataset, feature_name)
            feature_ds = feature_ds.map(self.adapted_preprocessors[feature_name])
            _preprocessor.adapt(feature_ds)
            encoded_feature = _preprocessor(feature_input)
            encoded_features[feature_name] = encoded_feature
            self.adapted_preprocessors[feature_name] = _preprocessor
        return encoded_features

    def __getitem__(self, idx):
        return self.feature_encoder_list[idx]

    def __len__(self):
        return len(self.feature_encoder_list)

    @property
    def encoder_name(self):
        return self[0][0]


class _BaseSingleEncoder(BaseEncoder):
    """
    Preprocess each feature based on specified preprocessing layer contained in feature_encoder_list
    """
    def __init__(self, feature_encoder_list=None):
        super().__init__(feature_encoder_list=feature_encoder_list)

    def encode(self, dataset):
        """Apply feature encodings on supplied feature encoding list

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, dict): Keras inputs for each feature and dict of encoders
        """
        name, preprocessor, features = self.feature_encoder_list
        feature_inputs = self.create_inputs(features, preprocessor.dtype)
        encoded_features = self._encode_one(dataset, preprocessor, features, feature_inputs)
        return feature_inputs, encoded_features


class _BaseMultipleEncoder(BaseEncoder):
    """
    Preprocessing pipeline to apply multiple encoders in serie
    """
    def __init__(self, feature_encoder_list=None):
        super().__init__(feature_encoder_list=feature_encoder_list)

    def encode(self, dataset):
        """Apply feature encodings on supplied feature encoding list

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            (list, dict): Keras inputs for each feature and dict of encoders
        """
        name, preprocessor, features = self.feature_encoder_list[0]
        feature_inputs = self.create_inputs(features, preprocessor.dtype)
        encoded_features = self._encode_one(dataset, preprocessor, features, feature_inputs)
        if len(self.feature_encoder_list) == 1:
            # _BaseMultipleEncoder use case is for multiple encoders applied on the same features
            # It should never have only one encoder. Adding this step for robustness
            return feature_inputs, encoded_features
        for (name, preprocessor, features) in self.feature_encoder_list[1:]:
            encoded_features = self._encode_one(dataset, preprocessor, features, [v for v in encoded_features.values()])
        return feature_inputs, encoded_features
