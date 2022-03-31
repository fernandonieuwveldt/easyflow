"""base classes for stateful preprocessing layers"""
from abc import ABC, abstractmethod
import tensorflow as tf

from custom import Pipeline


def one2one_func(x):
    """helper method to apply one to one preprocessor"""
    return x


def extract_feature_column(dataset, name):
    feature = dataset.map(lambda x, y: x[name])
    feature = feature.map(lambda x: tf.expand_dims(x, -1))
    return feature


class FeatureProcessor(tf.keras.layers.Layer):
    def __init__(self, feature_preprocessor_list=[], *args, **kwargs):
        super(FeatureProcessor, self).__init__(*args, *kwargs)
        self.feature_preprocessor_list = feature_preprocessor_list
        self.adapted_preprocessors = {}

    def adapt(self, dataset):
        for _, preproc_steps, features in self.feature_preprocessor_list:
            # get initial preprocessing layer config
            config = preproc_steps.get_config()
            for feature in features:
                config.pop("name", None)
                # get a fresh preprocessing instance
                cloned_preprocessor = preproc_steps.from_config(config)
                feature_ds = extract_feature_column(dataset, feature)
                # check if layer has adapt method
                cloned_preprocessor.adapt(feature_ds)
                self.adapted_preprocessors[feature] = cloned_preprocessor

    def call(self, inputs):
        forward_pass_list = []
        for _, _, features in self.feature_preprocessor_list:
            forward_pass_list.extend(
                [
                    self.adapted_preprocessors[feature](inputs[feature])
                    for feature in features
                ]
            )
        return tf.keras.layers.concatenate(forward_pass_list)


if __name__ == "__main__":
    import pandas as pd
    from easyflow.data.mapper import TensorflowDataMapper

    import tensorflow as tf
    from tensorflow.keras.layers.experimental.preprocessing import (
        Normalization,
        IntegerLookup,
        StringLookup,
    )

    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    dataframe = pd.read_csv(file_url)
    labels = dataframe.pop("target")

    batch_size = 32
    dataset_mapper = TensorflowDataMapper()
    dataset = dataset_mapper.map(dataframe, labels)
    train_data_set, val_data_set = dataset_mapper.split_data_set(dataset)
    train_data_set = train_data_set.batch(batch_size)
    val_data_set = val_data_set.batch(batch_size)

    NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]
    CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "ca"]
    # thal is represented as a string
    STRING_CATEGORICAL_FEATURES = ["thal"]

    dtype_mapper = {
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
        "thal": tf.string,
    }

    def create_inputs(data_type_mapper):
        """Create model inputs
        Args:
            data_type_mapper (dict): Dictionary with feature as key and dtype as value
                                    For example {'age': tf.float32, ...}
        Returns:
            (dict): Keras inputs for each feature
        """
        return {
            feature: tf.keras.Input(shape=(1,), name=feature, dtype=dtype)
            for feature, dtype in data_type_mapper.items()
        }

    feature_encoder_list = [
        ("numeric_encoder", Normalization(), NUMERICAL_FEATURES),
        (
            "categorical_encoder",
            IntegerLookup(output_mode="binary"),
            CATEGORICAL_FEATURES,
        ),
        # # # For feature thal we first need to run StringLookup followed by a IntegerLookup layer
        (
            "string_encoder",
            Pipeline([StringLookup(), IntegerLookup(output_mode="binary")]),
            STRING_CATEGORICAL_FEATURES,
        ),
    ]

    preproc = FeatureProcessor(feature_encoder_list)
    preproc.adapt(train_data_set)

    feature_layer_inputs = create_inputs(dtype_mapper)
    preprocessed_data = preproc(feature_layer_inputs)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(preprocessed_data)
    # we need a class or method that returns a preprocessing submodel
    model = tf.keras.Model(inputs=feature_layer_inputs, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.fit(train_data_set, epochs=10)
