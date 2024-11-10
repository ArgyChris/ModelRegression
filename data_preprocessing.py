from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Optional, List
import numpy as np
import pandas as pd
import kagglehub
import os

class DataPreprocessing:
    """
    Data processing utility for downloading and preprocessing: imputation, one-hot encoding, feature engineering, sampling
    """
    @staticmethod
    def dataset_download(dataset_path: str, file_name: str = "housing.csv") -> pd.DataFrame:
        """
        Download dataset using Kaggle hub API and load specified CSV file.
        
        Args:
            dataset_path (str): Kaggle dataset path in the format 'user/dataset'.
            file_name (str): CSV file name to load from the downloaded dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset as a DataFrame.
        
        Raises:
            FileNotFoundError: If the specified file is not found in the dataset.
        """
        path = kagglehub.dataset_download(dataset_path)
        data_path = os.path.join(path, file_name)
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            raise FileNotFoundError(f"{file_name} not found in dataset {dataset_path}")

    class Imputator(BaseEstimator, TransformerMixin):
        """
        Custom imputer class to apply imputation only to numerical columns
        """
        def __init__(self, strategy: str = 'median'):
            """
            Initialize the imputer with the specified strategy.
            
            Args:
                strategy (str): Imputation strategy ('mean', 'median', or 'most_frequent').
            """
            self.strategy = strategy
            self.imputer = SimpleImputer(strategy=self.strategy)
        
        def fit(self, X: pd.DataFrame, y=None):
            """
            Fit the imputer only to the numerical columns
            
            Args:
                X (pd.DataFrame): Input DataFrame with numerical columns.
                y: Ignored.
                
            Returns:
                self
            """
            self.imputer.fit(X.select_dtypes(include=['float64', 'int64']))
            return self
        
        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            """
            Impute missing values in numerical columns based on the strategy.
            
            Args:
                X (pd.DataFrame): Input DataFrame with numerical columns.
                
            Returns:
                pd.DataFrame: Transformed DataFrame with imputed values.
            """
            X_transformed = X.copy()
            X_transformed[X.select_dtypes(include=['float64', 'int64']).columns] = self.imputer.transform(X.select_dtypes(include=['float64', 'int64']))
            return X_transformed
        
    class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
        """
        Custom one-hot encoder for specified categorical columns, with special handling for predictions.
        """
        def __init__(self, columns: List[str], prediction: Optional[bool] = False):
            """
            Initialize the encoder with specified columns and mode (training or prediction).
            
            Args:
                columns (List[str]): List of categorical columns to encode.
                prediction (bool, optional): Flag for prediction mode (default: False).
            """
            self.columns = columns  # List of categorical columns to apply one-hot encoding
            self.prediction = prediction

        def fit(self, X: pd.DataFrame, y=None):
            # No fitting is necessary for one-hot encoding, we only need the columns specified
            return self
        
        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            """
            Apply one-hot encoding to specified columns.
            
            Args:
                X (pd.DataFrame): Input DataFrame with numerical and a categorical column.
                
            Returns:
                pd.DataFrame: Transformed DataFrame with replacement of the categorical column with one-hot encoded columns.
            """
            X_transformed = X.copy()
            if not self.prediction:
                # Apply pd.get_dummies only to the specified categorical columns
                X_transformed = pd.get_dummies(X_transformed, columns=self.columns)
                
                # Find the newly created dummy columns and convert them to integers (0 or 1)
                dummy_columns = [col for col in X_transformed.columns if any(col.startswith(c) for c in self.columns)]
                X_transformed[dummy_columns] = X_transformed[dummy_columns].astype(int)
            else:
                # Manual one-hot encoding 
                one_hot_vect = np.eye(5)[np.array([int(X_transformed[self.columns].values[0][0])-1]).reshape(-1)]
                one_hot_df = pd.DataFrame(one_hot_vect, columns=[
                    'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
                    'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'
                ])
                X_transformed = X_transformed.join(one_hot_df)
                X_transformed.drop('ocean_proximity', axis=1, inplace=True)
        
            return X_transformed
        
    class FeatureEngineering(BaseEstimator, TransformerMixin):
        """
        Custom feature engineering transformer to create new derived features.
        """
        def __init__(self, add_population_density: bool = True, add_bedrooms_per_room: bool = True, add_rooms_per_household: bool = True, add_income_categories: bool = True):
            self.add_population_density = add_population_density
            self.add_bedrooms_per_room = add_bedrooms_per_room
            self.add_rooms_per_household = add_rooms_per_household
            self.add_income_categories = add_income_categories
        
        def fit(self, X: pd.DataFrame, y=None):
            return self
        
        def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
            """
            Apply feature engineering to the input data.
            
            Args:
                X (pd.DataFrame): Input DataFrame.
                
            Returns:
                pd.DataFrame: Transformed DataFrame with the additionally derived features.
            """
            X_transformed = X.copy()
            first_ocean_proximity_col = [col for col in X_transformed.columns if col.startswith('ocean_proximity')][0]
            insertion_index = X_transformed.columns.get_loc(first_ocean_proximity_col)

            # Calculate new features and insert them at the identified position
            if self.add_population_density:
                X_transformed.insert(insertion_index, 'population_density', X_transformed['population'] / X_transformed['households'])
                insertion_index+=1
            if self.add_bedrooms_per_room:
                X_transformed.insert(insertion_index, 'bedrooms_per_room', X_transformed['total_bedrooms'] / X_transformed['total_rooms'])
                insertion_index+=1
            if self.add_rooms_per_household:     
                X_transformed.insert(insertion_index, 'rooms_per_household', X_transformed['total_rooms'] / X_transformed['households'])
                insertion_index+=1
            if self.add_income_categories:
                X_transformed.insert(insertion_index, 'income_categories', np.ceil(X_transformed["median_income"] / 1.5))
                X_transformed['income_categories'].where(X_transformed['income_categories'] < 5, 5.0, inplace=True)
            
            return X_transformed
        
    class StratifiedSplitter(BaseEstimator, TransformerMixin):
        """
        Custom stratified splitter to split data into training and testing sets based on a stratification column.
        """
        def __init__(self, stratify_col: str, test_size: float = 0.2, random_state: int = 42):
            self.stratify_col = stratify_col
            self.test_size = test_size
            self.random_state = random_state

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):        
            """
            Perform a stratified split on the data based on the chosen column.
            
            Args:
                X (pd.DataFrame): Input DataFrame.
                
            Returns:
                Tuple: Training features, test features, training labels, test labels.
            """
            split = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            for train_index, test_index in split.split(X, X[self.stratify_col]):
                strat_train_set = X.loc[train_index]
                strat_test_set = X.loc[test_index]
            
            # Separate features and labels
            strat_train_label = strat_train_set['median_house_value']
            strat_train_set = strat_train_set.drop(columns=['median_house_value', self.stratify_col])
            
            strat_test_label = strat_test_set['median_house_value']
            strat_test_set = strat_test_set.drop(columns=['median_house_value', self.stratify_col])
            
            return strat_train_set, strat_test_set, strat_train_label, strat_test_label
