from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Optional, List
import numpy as np
import os
import joblib

class CustomModelTraining(BaseEstimator, RegressorMixin):
    """
    A custom model training class to initialize and train either a Linear Regression 
    or Decision Tree Regressor model, with optional scaling support.
    """
    def __init__(self, model_type: str = 'linear', random_state: int = 42, scaler: Optional[object] = None):
        """
        Initialize the model with specified type, random state, and scaler.
        
        Args:
            model_type (str): Type of model to use ('linear' or 'tree').
            random_state (int): Seed for reproducibility (used in DecisionTreeRegressor).
            scaler (Optional[object]): Optional scaler to inverse transform predictions.
        
        Raises:
            ValueError: If an invalid model type is provided.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = scaler

        #  Select the model based on the model_type parameter
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'tree':
            self.model = DecisionTreeRegressor(random_state=self.random_state)
        else:
            raise ValueError("Model type should be 'linear' or 'tree'")

    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit the selected model on the data.

        Args:
            X (np.ndarray): Feature matrix.
            Y (np.ndarray): Target vector.
        
        Returns:
            CustomModelTraining: Trained model instance.
        """
        self.model.fit(X, Y)
        return self
    
    def predict(self, X: np.ndarray):
        """
        Predict using the selected model.

        Args:
            X (np.ndarray): Feature matrix for predictions.
        
        Returns:
            Union[np.ndarray, list]: Predicted values, optionally inverse transformed if scaler is set.
        """
        predictions = self.model.predict(X)
        
        # Inverse transform the predictions if a scaler is provided
        if self.scaler:
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        
        return predictions
    
class ModelSaver:
    """
    A utility class to save the trained model and scalers to disk.
    """
    def __init__(self, model: Optional[object] = None, scalerX: Optional[object] = None, scalerY: Optional[object] = None, path: Optional[object] = None):
        """
        Initialize ModelSaver with model, scalers, and save path.
        
        Args:
            model (Optional[object]): Trained model to save.
            scalerX (Optional[object]): Scaler used for input features.
            scalerY (Optional[object]): Scaler used for target values.
            path (Optional[str]): Directory path to save model and scalers (default: current directory).
        """
        self.model = model
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.path = path if path is not None else './'

    def save(self):
        """
        Saves the models and scalers to disck
        """
        joblib.dump(self.model, os.path.join(self.path, 'model.pkl'))
        print(f"Model saved to {self.path}")

        joblib.dump(self.scalerX, os.path.join(self.path, 'scaler_X.pkl'))
        print(f"ScalerX saved to {self.path}")
        
        joblib.dump(self.scalerY, os.path.join(self.path, 'scaler_Y.pkl'))
        print(f"ScalerY saved to {self.path}") 

class MetricsEvaluator:
    """
    A class to evaluate model performance using various regression metrics, including
    RMSE, R2, MSE, and MAE. Optionally, a scatter plot of actual vs. predicted values can be generated.
    """
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Initialize the metrics evaluator.
        
        Args:
            y_true (np.ndarray): Actual target values.
            y_pred (np.ndarray): Predicted target values.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def rmse(self) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE).

        Returns:
            float: The RMSE of the predictions.
        """
        mse = mean_squared_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mse)
        return rmse

    def r2(self) -> float:
        """
        Calculate R2 score (coefficient of determination).

        Returns:
            float: The R2 score of the predictions.
        """
        return r2_score(self.y_true, self.y_pred)

    def mse(self) -> float:
        """
        Calculate Mean Squared Error (MSE).

        Returns:
            float: The MSE of the predictions.
        """
        return mean_squared_error(self.y_true, self.y_pred)

    def mae(self) -> float:
        """
        Calculate Mean Absolute Error (MAE).

        Returns:
            float: The MAE of the predictions.
        """
        return mean_absolute_error(self.y_true, self.y_pred)

    def __str__(self) -> str:
        """
        Return a string with the formatted evaluation metrics.

        Returns:
            str: Formatted string of RMSE, R2, MSE, and MAE.
        """
        metrics = {'RMSE': self.rmse(), 'R2': self.r2(), 'MSE': self.mse(), 'MAE': self.mae()}
        return f"RMSE: {metrics['RMSE']:.3f}, R2: {metrics['R2']:.3f}, MSE: {metrics['MSE']:.3f}, MAE: {metrics['MAE']:.3f}"

    def plot_predictions(self):
        """
        Plot a scatter plot of predicted vs actual values.
        The plot includes a line for perfect predictions (y=x).
        """
        plt.figure(figsize=(8,6))
        plt.scatter(self.y_true, self.y_pred, color='blue', label='Predictions vs Actuals')
        plt.plot([min(self.y_true), max(self.y_true)], 
                 [min(self.y_true), max(self.y_true)], 
                 color='red', linestyle='--')  # Line of perfect prediction
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        plt.show()