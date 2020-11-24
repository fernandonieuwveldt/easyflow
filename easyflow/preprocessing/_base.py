import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


def _extract_feature_column(dataset, name):
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
        self.encoder.adapt(feature_dataset)
        return self

    def encode(self, input_feature, name, dataset):
        """
        """
        feature_ds = _extract_feature_column(dataset, name)
        encoded_feature = self.encoder(input_feature)
        return encoded_feature


class NormalizationEncoder(BaseFeatureEncoder):
    def __init__(self):
        super().__init__(Normalization())


class CategoricalEncoder(BaseFeatureEncoder):
    def __init__(self):
        super().__init__(CategoryEncoding(max_tokens=30, output_mode="binary"))


class StringIndexer(BaseFeatureEncoder):
    def __init__(self):
        super().__init__(StringLookup())


class StringCategoricalEncoder:
    def __init__(self):
        pass

    def encode(self, input_feature, name, dataset):
        """
        """
        feature_ds = _extract_feature_column(dataset, name)
        # apply String Indexer
        index_encoder = StringIndexer()
        index_encoder.adapt(feature_ds)
        index_encoded_feature = index_encoder.encode(input_feature, name, dataset)
        feature_ds = feature_ds.map(index_encoder.encoder)
        # apply categorical encoding
        category_encoder = CategoryEncoding(max_tokens=30, output_mode="binary")
        category_encoder.adapt(feature_ds)
        encoded_feature = category_encoder(index_encoded_feature)
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
        """
        Create inputs for Keras Model
        """
        return [keras.Input(shape=(1,), name=feature, dtype=self.dtype) for feature in self.features]

    def encode_input_features(self, all_inputs=None, dataset=None):
        """
        Encode Input with specified preprocessing layer
        """
        return [self.feature_encoder.encode(all_inputs[k], feature, dataset)\
            for (k, feature) in enumerate(self.features)]


class NormalizationInputEncoder(BaseKerasInput):
    """
    Normalization encoder for numerical features
    """
    def __init__(self, features):
        super().__init__(features=features,
                         feature_encoder=NormalizationEncoder(),
                         dtype="float64")


class CategoricalInputEncoder(BaseKerasInput):
    """
    One hot encoding for categorical features
    """
    def __init__(self, features):
        super().__init__(features=features,
                         feature_encoder=CategoricalEncoder(),
                         dtype="int64")


class StringInputEncoder(BaseKerasInput):
    """
    First creates string indexer than apply's one hot encoding
    """
    def __init__(self, features):
        super().__init__(features=features,
                         feature_encoder=StringCategoricalEncoder(),
                         dtype="string")


class StringInputIndexer(BaseKerasInput):
    """
    First creates string indexer than apply's one hot encoding
    """
    def __init__(self, features):
        super().__init__(features=features,
                         feature_encoder=StringIndexer(),
                         dtype="string")
