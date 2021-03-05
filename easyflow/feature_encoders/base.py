"""Base Classes for encoders using tensrflow feature columns"""
from abc import ABC, abstractmethod


class BaseFeatureColumnEncoder(ABC):
    """Base class for a tensorlow feature column based encoder"""

    @abstractmethod
    def encode(self, X):
        """Encoder to be implemented

        Args:
            X pandas.DataFrame: Features Data to apply encoder on.
        """
