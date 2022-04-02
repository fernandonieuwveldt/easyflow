import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import (
    Normalization,
    IntegerLookup,
    StringLookup,
)
from easyflow.data.mapper import TensorflowDataMapper
from easyflow.preprocessing.custom import FeatureInputLayer, PreprocessingChainer
from easyflow.preprocessing.pipeline import FeaturePreprocessor, FeaturePreprocessorUnion

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

# TODO: Each step can be a layer??? For concatenation later
feature_encoder_list = [
    ("numeric_encoder", Normalization(), NUMERICAL_FEATURES),
    (
        "categorical_encoder",
        IntegerLookup(output_mode="binary"),
        CATEGORICAL_FEATURES,
    ),
    (
        "string_encoder",
        PreprocessingChainer([StringLookup(), IntegerLookup(output_mode="binary")]),
        STRING_CATEGORICAL_FEATURES,
    ),
]

preproc = FeaturePreprocessorUnion(feature_encoder_list)
preproc.adapt(train_data_set)

feature_layer_inputs = FeatureInputLayer(dtype_mapper)
preprocessed_data = preproc(feature_layer_inputs)

# preproc = FeaturePreprocessor(feature_encoder_list)
# preproc.adapt(train_data_set)

# feature_layer_inputs = FeatureInputLayer(dtype_mapper)
# preprocessed_data = preproc(feature_layer_inputs)
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
