import tensorflow as tf
import autokeras as ak


class BaseAutoKerasModel:
    """
    Base class for AutoKeras models integrated with EasyFlow preprocessing pipeline.
    """
    def __init__(self, preprocessor, **kwargs):
        """
        Initialize the base AutoKeras model.

        Args:
            preprocessor: EasyFlow preprocessor instance.
            **kwargs: Additional keyword arguments passed to the AutoKeras model.
        """
        self.preprocessor = preprocessor
        self.kwargs = kwargs

    def _preprocessor_function(self, x, y=None):
        """
        Helper function to apply the EasyFlow preprocessing pipeline to the input dataset.

        Args:
            x: Input features.
            y: Target labels (optional).

        Returns:
            Preprocessed inputs and, if provided, target labels.
        """
        inputs = {name: tf.keras.Input(shape=(1,), name=name, dtype=dtype_mapper[name]) for name in dataframe.columns}
        preprocessed_inputs = self.preprocessor(inputs)
        if y is not None:
            return preprocessed_inputs, y
        return preprocessed_inputs

    def build_model(self):
        """
        Build the AutoKeras model. This method should be implemented by subclasses.

        Raises:
            NotImplementedError: If called on the base class.
        """
        raise NotImplementedError("Subclasses must implement the `build_model` method")

    def fit(self, train_data_set, val_data_set=None, **kwargs):
        """
        Train the AutoKeras model using the EasyFlow preprocessing pipeline.

        Args:
            train_data_set: Training dataset.
            val_data_set: Validation dataset (optional).
            **kwargs: Additional keyword arguments passed to the AutoKeras model's fit method.

        Returns:
            The training history object.
        """
        preprocessed_train_data_set = train_data_set.map(self._preprocessor_function)
        if val_data_set:
            preprocessed_val_data_set = val_data_set.map(self._preprocessor_function)
        else:
            preprocessed_val_data_set = None

        model = self.build_model()
        history = model.fit(preprocessed_train_data_set, validation_data=preprocessed_val_data_set, **kwargs)
        self.model = model
        return history

    def predict(self, dataset, **kwargs):
        """
        Make predictions using the trained AutoKeras model.

        Args:
            dataset: Dataset for making predictions.
            **kwargs: Additional keyword arguments passed to the AutoKeras model's predict method.

        Returns:
            Predictions.
        """
        preprocessed_dataset = dataset.map(self._preprocessor_function)
        return self.model.predict(preprocessed_dataset, **kwargs)

class AutoKerasClassifier(BaseAutoKerasModel):
    """
    AutoKeras classifier integrated with EasyFlow preprocessing pipeline.
    """
    def build_model(self):
        """
        Build the AutoKeras classifier.

        Returns:
            The AutoKeras classifier model.
        """
        model = ak.StructuredDataClassifier(inputs=self._preprocessor_function(None), **self.kwargs)
        return model

class AutoKerasRegressor(BaseAutoKerasModel):
    """
    AutoKeras regressor integrated with EasyFlow preprocessing pipeline.
    """
    def build_model(self):
        """
        Build the AutoKeras regressor.

        Returns:
            The AutoKeras regressor model.
        """
        model = ak.StructuredDataRegressor(inputs=self._preprocessor_function(None), **self.kwargs)
        return model
