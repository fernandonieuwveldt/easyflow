"""Preprocessing pipelines 
"""

from easyflow.preprocessing.pipeline import (
    FeaturePreprocessor,
    FeaturePreprocessorUnion,
)
from easyflow.preprocessing.custom import (
    FeatureInputLayer,
    NumericPreprocessingLayer,
    SequentialPreprocessingChainer,
)
# This will be removed from later versions. Raises exception when initialised
from easyflow.preprocessing.pipeline import (
    Pipeline,
    FeatureUnion,
)


__all__ = [
    "FeatureInputLayer",
    "FeaturePreprocessor",
    "FeaturePreprocessorUnion",
    "NumericPreprocessingLayer",
    "SequentialPreprocessingChainer",
    # Below will be removed
    "Pipeline",
    "FeatureUnion"
]
