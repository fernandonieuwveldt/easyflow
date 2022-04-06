"""base class and usefull methods"""
from easyflow.preprocessing.custom import NumericPreprocessingLayer


class BaseFeaturePreprocessor:
    """Interface for all Feature Preprocessor
    """
    def map_preprocessor(self, steps):
        """Check and Map input if any of the preprocessors are None, i.e. use as is. For 
        example Binary features that don't need further preprocessing   

        Args:
            steps (list): Preprocessing steps.

        Returns:
            list: List of mapped preprocessors if None was supplied.
        """
        selector = lambda _preprocessor: _preprocessor or NumericPreprocessingLayer()
        return [
            (name, selector(preprocessor), step) for name, preprocessor, step in steps
        ]

    def __getitem__(self, idx):
        # This should rather return the adapted layers for the specific step
        return self.feature_preprocessor_list[idx]

    def __len__(self):
        """Total number of steps

        Returns:
            int: Total number of steps
        """
        return len(self.feature_preprocessor_list)

    @property
    def preprocessor_name(self):
        """Return the step names

        Returns:
            list: List of step names
        """
        return [self[k][0] for k in range(len(self))]

    @classmethod
    def from_infered_pipeline(cls, dataset):
        """Infer standard pipeline for structured data, i.e Normalization for numerical
        features and StringLookup/IntegerLookup for categoric features

        Args:
            dataset (tf.data.Dataset): Features Data to apply encoder on.

        Returns:
            BaseFeaturePreprocessorLayer: Initilized BaseFeaturePreprocessorLayer object
        """
        raise NotImplementedError
