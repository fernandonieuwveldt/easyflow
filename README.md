Easy Tensorflow:

An interface containing easy tensorflow model building blocks and feature pipelines

## To install package:
```
pip install easy-tensorflow
```

# Preprocessing Encoder, Pipeline, SequentialEncoder and FeatureUnion example
The easyflow.preprocessing module contains functionality similar to what sklearn does with its Pipeline, FeatureUnion and ColumnTransformer does. Full example also in a notebook: easyflow/notebooks/preprocessing_example.ipynb

```
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization, CategoryEncoding, StringLookup

# local imports
from easyflow.data.mapper import TensorflowDataMapper
from easyflow.preprocessing.preprocessor import Encoder, Pipeline, SequentialEncoder, FeatureUnion
from easyflow.preprocessing.custom import IdentityPreprocessingLayer
```

### Read in data and map as tf.data.Dataset
Use the TensorflowDataMapper class to map pandas data frame to a tf.data.Dataset type.

```
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)
dataframe = dataframe.copy()
labels = dataframe.pop("target")

batch_size = 32
dataset_mapper = TensorflowDataMapper() 
dataset = dataset_mapper.map(dataframe, labels)
train_data_set, val_data_set = dataset_mapper.split_data_set(dataset)
train_data_set = train_data_set.batch(batch_size)
val_data_set = val_data_set.batch(batch_size)
```

### Set constants
```
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca']
# thal is represented as a string
STRING_CATEGORICAL_FEATURES = ['thal']
```

### Setup Preprocessing layer using FeatureUnion
Use Encoder and SequentialEncoder to preprocess features by putting everything in a FeatureUnion object.

```
feature_encoder_list = [
                        Encoder([('numeric_encoder', Normalization, NUMERICAL_FEATURES)]),
                        Encoder([('categorical_encoder', CategoryEncoding, CATEGORICAL_FEATURES)]),
                        # For feature thal we first need to run StringLookup followed by a CategoryEncoding layer
                        SequentialEncoder([('string_encoder', StringLookup, STRING_CATEGORICAL_FEATURES),
                                           ('categorical_encoder', CategoryEncoding, STRING_CATEGORICAL_FEATURES)])
                        ]

encoder = FeatureUnion(feature_encoder_list)
all_feature_inputs, preprocessing_layer = encoder.encode(dataset)
```

```
history=model.fit(train_data_set, validation_data=val_data_set, epochs=10)
```

# Model building Pipeline using easyflow feature_encoders module
This module is a fusion between keras layers and tensorflow feature columns. Full example also in a notebook: easyflow/notebooks/feature_column_example.ipynb

```
import pandas as pd
import tensorflow as tf

# local imports
from easyflow.data.mapper import TensorflowDataMapper
from easyflow.feature_encoders.transformer import FeatureColumnTransformer, FeatureUnionTransformer
from easyflow.feature_encoders.feature_encoder import NumericalFeatureEncoder, EmbeddingFeatureEncoder, CategoricalFeatureEncoder
```

### Load data
```
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

print(f"Train dataset shape: {data_frame.shape}")
```

### Map data frame to tf.data.Dataset

```
batch_size = 256
dataset_mapper = TensorflowDataMapper() 
dataset = dataset_mapper.map(data_frame, labels_binary)

train_data_set, val_data_set = dataset_mapper.split_data_set(dataset)
train_data_set = train_data_set.batch(batch_size)
val_data_set = val_data_set.batch(batch_size)
```

### Set up the feature encoding list
```
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
                        ('embedding_features_deep', EmbeddingFeatureEncoder(), EMBEDDING_FEATURES_NAMES),
                        ('embedding_features_wide', CategoricalFeatureEncoder(), EMBEDDING_FEATURES_NAMES)]
```

### Setting up feature layer and feature encoders
There are two main column transformer classes namely FeatureColumnTransformer and FeatureUnionTransformer. For this example we are going to build a Wide and Deep model architecture. So we will be using the FeatureColumnTransformer since it gives us more flexibility. FeatureUnionTransformer concatenates all the features in the input layer

```
feature_layer_inputs, feature_encoders =  FeatureColumnTransformer(feature_encoder_list).transform(train_data_set)
```

```
deep_features = feature_encoders['numerical_features']+\
                feature_encoders['categorical_features']+\
                feature_encoders['embedding_features_deep']

wide_features = feature_encoders['embedding_features_wide']
```

###  Set up Wide and Deep model architecture
```
deep = tf.keras.layers.DenseFeatures(deep_features)(feature_layer_inputs)
deep = tf.keras.layers.BatchNormalization()(deep)

wide = tf.keras.layers.DenseFeatures(wide_features)(feature_layer_inputs)

for nodes in [128, 64, 32]:
    deep = tf.keras.layers.Dense(nodes, activation='relu')(deep)
    deep = tf.keras.layers.Dropout(0.5)(deep)

wide_and_deep = tf.keras.layers.concatenate([deep, wide])
output = tf.keras.layers.Dense(1, activation='sigmoid')(wide_and_deep)
model = tf.keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=output)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])
```

Fit model
```
model.fit(train_data_set, validation_data=val_data_set, epochs=10)
```