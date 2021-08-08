## VERSION 1.1.1
* Fix README example 2

## VERSION 1.1.0
* Added EmbeddingCrossingFeatureEncoder
* CI/CD pipeline using github actions and publish to PYPI

## VERSION 1.0.0
* Update tensorflow to latest 2.5
* Major refactoring of feature encoders module
    - Refactor and added category crossing and bucketized encoders
    - Gixed vocab string types from bytes to strings to solve model saving issue
    - Added examples to class docstrings
* Major refactoring of preprocessing classes:
    - Added validator to base class
    - Fixed breaking changes when upgrading tf
    - Encapsulate Encoder class in pipeline classes
* Updated notebook examples and added saving and loading of models for end to end process.
* Updated docstrings
* Updated unittest
* Update README.md5

## VERSION 0.1.8
* Added LICENSE

## VERSION 0.1.7
* Added initialised preprocessing layers with args to encoder classes
* Added unit tests and test data for preprocessing and feature encoder modules
* CategoricalFeatureEncoder reads dtype straight from tensorflow dataset
* CHANGELOG.md file added

## VERSION 0.1.4
* Initial release 
* Tensorflow feature columns and Keras fusion with feature pipelines
* Feature Pipelines for Keras Preprocessing Layers
* tf.data.Dataset mapper from pandas DataFrame
