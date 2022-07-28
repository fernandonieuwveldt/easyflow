# EasyFlow: Keras Feature Preprocessing Pipelines

![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

# Table of Contents
1. [About EasyFlow](#about-EasyFlow)
2. [Motivation](#motivation)
3. [Installation](#installation)
4. [Example](#example)
5. [Tutorials](#tutorials)

---

## About EasyFlow

The `EasyFlow` package implements an interface similar to SKLearn's Pipeline API that contains easy feature preprocessing pipelines to build a full training and inference pipeline natively in Keras. All pipelines are implemented as Keras layers. 

---

## Motivation

There is a need to have a similar interface for Keras that mimics the SKLearn Pipeline API such as `Pipeline`, `FeatureUnion` and `ColumnTransformer`, but natively in Keras as Keras layers. The usual design pattern especially for tabular data is to first do preprocessing with SKLearn and then feed the data to a Keras model. With `EasyFlow` you don't need to leave the Tensorflow/Keras ecosystem to build custom pipelines and your preprocessing pipeline is part of your model architecture.

Main interfaces are:

* `FeaturePreprocessor`: This layer applies feature preprocessing steps and returns a separate layer for each step supplied. This gives more flexibility to the user and if a more advance network architecture is needed. For example something like a Wide and Deep network.
* `FeatureUnion`: This layer is similar to `FeaturePreprocessor` with an extra step that concatenates all layers into a single layer.

---

## Installation:

```bash
pip install easy-tensorflow
```

---

## Example

Lets look at a quick example:

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

This is the main part where `EasyFlow` fits in. We can now easily setup a feature preprocessing pipeline as a Keras layer with only a few lines of code.

```python
feature_preprocessor_list = [
    ('numeric_encoder', Normalization(), NUMERICAL_FEATURES),
    ('categorical_encoder', IntegerLookup(output_mode='multi_hot'), CATEGORICAL_FEATURES),
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

---

## Tutorials

### Migrate an Sklearn training Pipeline to Tensorflow Keras: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fernandonieuwveldt/easyflow/blob/develop/examples/migrating_from_sklearn_to_keras/migrate_sklearn_pipeline.ipynb)
* In this notebook we look at ways to migrate an Sklearn training pipeline to Tensorflow Keras. There might be a few reasons to move from Sklearn to Tensorflow.


### Single Input Multiple Output Preprocessor: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fernandonieuwveldt/easyflow/blob/develop/examples/single_input_multiple_output/single_input_multiple_output_preprocessor.ipynb)
* In this example we will show case how to apply different transformations and preprocessing steps on the same feature. What we have here is an example of a Single input Multiple output feature transformation scenario.

### Preprocessing module quick intro: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fernandonieuwveldt/easyflow/blob/develop/examples/preprocessing_example/preprocessing_example.ipynb)
* The `easyflow.preprocessing` module contains functionality similar to what Sklearn does with its `Pipeline`, `FeatureUnion` and `ColumnTransformer` does. This is a quick introduction.


### Tensorflow Feature columns quick intro: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fernandonieuwveldt/easyflow/blob/develop/examples/feature_column_demo/feature_column_example.ipynb)
*  Model building Pipeline using `EasyFlow` feature_encoders module. This module is a fusion between Keras layers and Tensorflow feature columns.
