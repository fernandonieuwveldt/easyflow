## VERSION 1.4.0
* Added MultiOutputTransformer that takes in one feature and applying mulitple transformations.
* Added an example notebook showcasing MultiOutputTransformer
* Changed description of project.

## VERSION 1.3.1
* Added StringToIntegerLookup to custom layer. This layer is a common usecase
* Refactored BaseFeaturePreprocessorLayer into a factory
* New class added for Tensorflow Datasets
* Support for Pandas in the EasyFlow pipeline
* Updated README and removed feature_columns example
* FeatureInputLayer is now a class and added init from dataset
* Bumped TF version from 2.7.0 to 2.7.1

## VERSION 1.3.0
* Refactor by implementing best practices from Keras and Tensorflow
* Layer's subclass layer class and implemented new custom layers
    - FeatureInputLayer: Dict of Input layers
    - PreprocessorChain: Adapt's multiple preprocessing layers
                            (Subclasses Layer)
    - PreprocessorChain: Adapt's multiple preprocessing layers
                                      (Subclasses Sequential model)
* Main interface is now subclassing Layer class:
    - BaseFeaturePreprocessorLayer: This class also implements adapt method
* New interfaces:
    - FeaturePreprocessor: Outputs a layer for each step
    - FeatureUnion: Concatenates steps into single layer
* Deprecated previous class Pipeline and FeatureUnion

## VERSION 1.2.0
* Added infered preprocessing Pipeline and FeatureUnion
* Updated README

## VERSION 1.1.3
* Fixed tensorflow and keras version conflicts

## VERSION 1.1.2
* Bump to tensorflow 2.6.0
* Apply tf 2.6.0 fixes
* Changed custom numeric preprocessing layer name from IdentityPreprocessingLayer to NumericPreprocessingLayer
* remove tf.data experimental unique

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
