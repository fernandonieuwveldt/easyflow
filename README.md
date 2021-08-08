Easy Tensorflow:

An interface containing easy tensorflow model building blocks and feature encoding pipelines

Model file structure:
```bash
├── easyflow
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── mapper.py
│   ├── feature_encoders
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── categorical_encoders.py
│   │   ├── numerical_encoders.py
│   │   └── pipeline.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── custom.py
│   │   ├── pipeline.py
│   └── tests
│       ├── __init__.py
│       ├── test_data
│       │   └── heart.csv
│       ├── test_feature_encoders.py
│       └── test_preprocessing.py
├── notebooks
│   ├── feature_column_example.ipynb
│   └── preprocessing_example.ipynb
├── CHANGELOG.md
├── LICENSE
├── MANIFEST.in
├── README.md
├── requirements.txt
└── setup.py
```

## To install package:
```bash
pip install easy-tensorflow
```

# Example 1: Preprocessing Pipeline and FeatureUnion example
The easyflow.preprocessing module contains functionality similar to what sklearn does with its Pipeline, FeatureUnion and ColumnTransformer does. Full example also in notebooks folder

```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization, StringLookup, IntegerLookup

# local imports
from easyflow.data import TensorflowDataMapper
from easyflow.preprocessing import FeatureUnion
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
```

### Setup Preprocessing layer using FeatureUnion

```python
feature_encoder_list = [
                        ('numeric_encoder', Normalization(), NUMERICAL_FEATURES),
                        ('categorical_encoder', IntegerLookup(output_mode='binary'), CATEGORICAL_FEATURES),
                        # For feature thal we first need to run StringLookup followed by a IntegerLookup layer
                        ('string_encoder', [StringLookup(), IntegerLookup(output_mode='binary')], STRING_CATEGORICAL_FEATURES)
                        ]

encoder = FeatureUnion(feature_encoder_list)
all_feature_inputs, preprocessing_layer = encoder.encode(dataset)
```

### Set up network
```python
# setup simple network
x = tf.keras.layers.Dense(128, activation="relu")(preprocessing_layer)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=all_feature_inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])

```

### Fit model
```python
history=model.fit(train_data_set, validation_data=val_data_set, epochs=10)
```

# Example 2: Model building Pipeline using easyflow feature encoders module
This module is a fusion between keras layers and tensorflow feature columns. 

FeatureColumnTransformer and FeatureUnionTransformer are the main interfaces and serves as feature transformation pipelines.

Wrapper classes exists for the following feature_columns
* CategoricalFeatureEncoder
* EmbeddingFeatureEncoder
* EmbeddingCrossingFeatureEncoder
* CategoryCrossingFeatureEncoder
* NumericalFeatureEncoder
* BucketizedFeatureEncoder

To create a custom encoder or one where wrapper class does not exist, there are two base interfaces to use:
* BaseFeatureColumnEncoder
* BaseCategoricalFeatureColumnEncoder

```python
import pandas as pd
import tensorflow as tf

# local imports
from easyflow.data import TensorflowDataMapper
from easyflow.feature_encoders import FeatureColumnTransformer, FeatureUnionTransformer
from easyflow.feature_encoders import NumericalFeatureEncoder, EmbeddingFeatureEncoder, CategoricalFeatureEncoder
```

### Load data
```python
CSV_HEADER = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_bracket",
]

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"


data_frame = pd.read_csv(data_url, header=None, names=CSV_HEADER)
labels = data_frame.pop("income_bracket")
labels_binary = 1.0 * (labels == " >50K")
data_frame.to_csv('adult_features.csv', index=False)
labels_binary.to_csv('adult_labels.csv', index=False)

```

### Map data frame to tf.data.Dataset

```python
batch_size = 256
dataset_mapper = TensorflowDataMapper() 
dataset = dataset_mapper.map(data_frame, labels_binary)

train_data_set, val_data_set = dataset_mapper.split_data_set(dataset)
train_data_set = train_data_set.batch(batch_size)
val_data_set = val_data_set.batch(batch_size)
```

### Set up the feature encoding list
```python
NUMERIC_FEATURE_NAMES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES_NAMES = [
    "workclass",
    "marital_status",
    "relationship",
    "race",
    "gender"]

EMBEDDING_FEATURES_NAMES = ['education',
                            'occupation',
                            'native_country']

feature_encoder_list = [('numerical_features', NumericalFeatureEncoder(), NUMERIC_FEATURE_NAMES),
                        ('categorical_features', CategoricalFeatureEncoder(), CATEGORICAL_FEATURES_NAMES),
                        ('embedding_features_deep', EmbeddingFeatureEncoder(dimension=10), EMBEDDING_FEATURES_NAMES),
                        ('embedding_features_wide', CategoricalFeatureEncoder(), EMBEDDING_FEATURES_NAMES)]
```

### Setting up feature layer and feature encoders
There are two main column transformer classes namely FeatureColumnTransformer and FeatureUnionTransformer. For this example we are going to build a Wide and Deep model architecture. So we will be using the FeatureColumnTransformer since it gives us more flexibility. FeatureUnionTransformer concatenates all the features in the input layer

```python
feature_layer_inputs, feature_layer =  FeatureColumnTransformer(feature_encoder_list).transform(train_data_set)
```

```python
deep = tf.keras.layers.concatenate([feature_layer['numerical_features'],
                                    feature_layer['categorical_features'],
                                    feature_layer['embedding_features_deep']])

wide = feature_layer['embedding_features_wide']
```

###  Set up Wide and Deep model architecture
```python
deep = tf.keras.layers.BatchNormalization()(deep)

for nodes in [128, 64, 32]:
    deep = tf.keras.layers.Dense(nodes, activation='relu')(deep)
    deep = tf.keras.layers.Dropout(0.5)(deep)

# combine wide and deep layers
wide_and_deep = tf.keras.layers.concatenate([deep, wide])
output = tf.keras.layers.Dense(1, activation='sigmoid')(wide_and_deep)
model = tf.keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=output)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])
```

Fit model
```python
model.fit(train_data_set, validation_data=val_data_set, epochs=10)
```
