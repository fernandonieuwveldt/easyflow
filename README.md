# EasyFlow:

Usually when we build ML training pipeline we might utilize `Pandas` to read and manipulate data; SKLearn to preprocess our features by using for example one-hot-encoding or Normalization. This can than be part of an `SKLearn Pipeline`. But what can you do if you need to implement a similar model training pipeline in `Keras`?

`EasyFlow` is a Keras and Tensorflow native implementation that mimics the functionality of SKLearn's Pipeline api, but implemented natively in Tensorflow and Keras. The `EasyFlow` package implements an interface that contains easy feature preprocessing pipelines to build a full training and inference pipeline by only utilizing the Tensorflow and Keras framework. A usecase could be if one needs to migrate an existing SKLearn training and inference pipeline to Tensorlow and Keras. Or if you need to make use of other benefits one can get with Tensorflow; for example Tensorflow serving to serve models or MLOps with TFX.

Generally we tend to use SKLearn Pipeline for feature engineering and preprocessing before feeding data to a Keras model. Here we end up with `multiple artifacts`; one for preprocessing and feature engineering from SKLearn and the other a Keras saved model. For this case the preprocessing is not part of Keras model which can cause `training-serving skew`. And from a `serving perspective the pipeline is also disconnected` and an extra step is needed to feed encoded data to the Keras model. If you want to use Tensorflow serving for serving models this can cause issues because the preprocessing depends on different python libary. So data first needs to preprocessed before sending for inference. Recently Keras implemented preprocessing layers. Using these layers the Data scientist can apply preprocessing layers as part of the neural network architecture which will prevent training-serving skew. 

One missing component in Keras is a `Pipeline` or `ColumnTransformer` type implementation for Keras preprocessing layers. The EasyFlow package implements these feature Pipelines with an easy interface as Feature Preprocessing layers. Main interfaces are:

* `FeaturePreprocessor`: This layer applies feature preprocessing steps and returns a separate layer for each       step supplied. This gives more flexibility to the user and if a more advance network architecture is needed. For example something like a Wide and Deep network.
* `FeatureUnion`: This layer is similar to `FeaturePreprocessor` with an extra step that concatenates all
layers into a single layer.

## Usage:

Chaining preprocessing layers

```python
def StringToIntegerLookup():
    return PreprocessorChain(
        [StringLookup(), IntegerLookup(output_mode='binary')]
    )
```
The `PreprocessorChain` can be use to chain multiple layers especially usefull when these steps are dependent on each other.

The `FeatureUnion` layer is one of the two interfaces. Note that we can use our layer above as one of the steps.

```python
# FeatureUnion is a Keras layer.
preprocessor = FeatureUnion([
    ('normalization', Normalization(), FEATURES_TO_NORMALIZE),
    ('one_hot_encoding', IntegerLookup(output_mode='binary'), FEATURES_TO_ENCODE),
    ('string_encoder', StringToIntegerLookup(), STR_FEATURES_TO_ENCODE)
])

# to update the states for preprocess layers:
preprocessor.adapt(data)
```

`Easyflow` also supports both `Pandas DataFrame's` and `tf.data.Dataset` types as input to the Feature Preprocessing pipelines.

(There is also a training pipeline module for Tensorflow feature columns. This will in future be deprecated and the focus will be on Keras preprocessing module.)

For more examples in future. Check out the python notebooks in the notebooks folder.

# Installation:
```bash
pip install easy-tensorflow
```

# Example: Preprocessing using FeatureUnion
The FeatureUnion interface is similar to SKLearn's ColumnTransformer. Full example also in notebooks folder.

```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Normalization, StringLookup, IntegerLookup

# local imports
from easyflow.data import TensorflowDataMapper
from easyflow.preprocessing import FeatureUnion
from easyflow.preprocessing import (
    FeatureInputLayer,
    StringToIntegerLookup,
)

```

### Read in data and map as tf.data.Dataset
Use the TensorflowDataMapper class to map pandas data frame to a tf.data.Dataset type.

```python
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)
labels = dataframe.pop("target")

batch_size = 32
dataset_mapper = TensorflowDataMapper() 
dataset = dataset_mapper.map(dataframe, labels)
train_data_set, val_data_set = dataset_mapper.split_data_set(dataset)
train_data_set = train_data_set.batch(batch_size)
val_data_set = val_data_set.batch(batch_size)
```

### Set constants
```python
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca']
# thal is represented as a string
STRING_CATEGORICAL_FEATURES = ['thal']

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
```

### Setup Preprocessing layer using FeatureUnion

```python
feature_preprocessor_list = [
    ('numeric_encoder', Normalization(), NUMERICAL_FEATURES),
    ('categorical_encoder', IntegerLookup(output_mode='binary'), CATEGORICAL_FEATURES),
    ('string_encoder', StringToIntegerLookup(), STRING_CATEGORICAL_FEATURES)
]

preprocessor = FeatureUnion(feature_preprocessor_list)
preprocessor.adapt(train_data_set)

feature_layer_inputs = FeatureInputLayer(dtype_mapper)
preprocessing_layer = preprocessor(feature_layer_inputs)
```

### Set up network
```python
# setup simple network
x = tf.keras.layers.Dense(128, activation="relu")(preprocessing_layer)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=feature_layer_inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])
```

### Fit model
```python
history=model.fit(train_data_set, validation_data=val_data_set, epochs=10)
```
