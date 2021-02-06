from tensorflow.keras.layers.experimental.preprocessing import Normalization

from .base import BaseFeatureEncoder, BaseKerasInput


class NormalizationEncoder(BaseFeatureEncoder):
    """
    Normalization encoder for numerical features
    """
    def __init__(self):
        super().__init__(Normalization())


class NormalizationInputEncoder(BaseKerasInput):
    """
    Normalization encoder for numerical features
    """
    def __init__(self, features):
        super().__init__(features=features,
                         feature_encoder=NormalizationEncoder(),
                         dtype="float64")
