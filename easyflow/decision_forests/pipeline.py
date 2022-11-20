import tensorflow_decision_forests as tfdf

import tensorflow as tf
from tensorflow.keras.layers import Normalization, IntegerLookup, StringLookup

from easyflow.data.mapper import TensorflowDataMapper
from easyflow.preprocessing.pipeline import FeatureUnion
from easyflow.preprocessing import FeatureInputLayer, StringToIntegerLookup


class RandomForestModel:
    """Create a model pipeline with an EasyFlow Pipeline Head.

    Args:
        preprocessor (_type_, optional): _description_. Defaults to None.
        model (_type_, optional): _description_. Defaults to None.
    """
    def __init__(preprocessor=None, *args, **kwargs):
        self.preprocessor = preprocessor
        self.model = tfdf.keras.RandomForestModel(
            preprocessing=self.preprocessor, *args, **kwargs)
        )
        self.is_adapted = None
    
    def adapt(self, dataset, *args, **kwargs):
        """_summary_

        Args:
            dataset (_type_): _description_
        """
        self.preprocessor.adapt(dataset, *args, **kwargs)
        self.is_adapted = True
    
    def fit(self, dataset, *args, **kwargs):
        self.model.



if __name__ == '__main__':
    feature_layer_inputs = FeatureInputLayer({
        "age": tf.float32,
        "sex": tf.float32,
        "cp": tf.float32,
        "trestbps": tf.float32,
        "chol": tf.float32,
        "fbs": tf.float32,
        "restecg": tf.float32,
        "thalach": tf.float32,
        "exang": tf.float32,
        "oldpeak": tf.float32,
        "slope": tf.float32,
        "ca": tf.float32,
        "thal": tf.string
    })


    preprocessor = FeatureUnion(
        feature_preprocessor_list = [
            ('num', Normalization(), NUMERICAL_FEATURES),
            ('cat', IntegerLookup(output_mode='binary'), CATEGORICAL_FEATURES),
            ('str_cat', StringToIntegerLookup(), STRING_CATEGORICAL_FEATURES)
        ]
    )

    # to update the states for preprocess layers:
    preprocessor.adapt(dataframe)
    preprocessed_inputs = preprocessor(feature_layer_inputs)
    preprocessing_model = tf.keras.Model(feature_layer_inputs, preprocessed_inputs)
