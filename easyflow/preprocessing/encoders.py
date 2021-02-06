from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from .base import extract_feature_column


class CategoricalEncoder(BaseFeatureEncoder):
    def __init__(self):
        super().__init__(CategoryEncoding(max_tokens=30, output_mode="binary"))


class StringIndexer(BaseFeatureEncoder):
    def __init__(self):
        super().__init__(StringLookup())


class StringCategoricalEncoder:
    def __init__(self, max_tokens=None):
        self.max_tokens = max_tokens

    def encode(self, input_feature, name, dataset):
        """
        """
        feature_ds = extract_feature_column(dataset, name)
        # apply String Indexer
        index_encoder = StringIndexer()
        index_encoder.adapt(feature_ds)
        index_encoded_feature = index_encoder.encode(input_feature, name, dataset)
        feature_ds = feature_ds.map(index_encoder.encoder)
        # apply categorical encoding
        category_encoder = CategoryEncoding(max_tokens=self.max_tokens, output_mode="binary")
        category_encoder.adapt(feature_ds)
        encoded_feature = category_encoder(index_encoded_feature)
        return encoded_feature


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
