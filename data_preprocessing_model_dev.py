import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_preprocessing import DataPreprocessing
from model_training import CustomModelTraining, MetricsEvaluator, ModelSaver

def train_eval_model(): 
    """
    Main module for training, evaluating, and saving machine learning models.
    """
    # Step 1: Download and load the dataset
    data = DataPreprocessing.dataset_download("camnugent/california-housing-prices")
    categorical_cols = ['ocean_proximity']

    #Step 2: Preprocessing pipeline definition and running
    data_preprocessing_pipeline = Pipeline([
        ('imputator', DataPreprocessing.Imputator(strategy='median')),  # Impute missing values with median using custom Imputator
        ('one_hot_encoder', DataPreprocessing.CustomOneHotEncoder(columns=categorical_cols)), #One-hot-encoding of the categorical features using custom One-hot encoder
        ('feature_eng', DataPreprocessing.FeatureEngineering(True, True, True, True)), #Feature engineering to include additional features
        ('stratified_split', DataPreprocessing.StratifiedSplitter(stratify_col='income_categories', test_size=0.2, random_state=42)) #Stratified sampling with respect to the income categories
    ])
    X_train, X_test, Y_train, Y_test = data_preprocessing_pipeline.fit_transform(data)

    #Step 3: Feature scaling
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    Y_train_scaled = scaler_Y.fit_transform(Y_train.to_numpy().reshape(-1, 1))

    #Step 4: Main training pipeline definition and evaluation
    model_linear_regression = Pipeline([
        ('model', CustomModelTraining(model_type='linear', scaler=scaler_Y))  #Model choice: Linear regression [linear] or Decision tree [tree]
    ])
    model_linear_regression.fit(X_train_scaled, Y_train_scaled)
    y_pred = model_linear_regression.predict(X_test_scaled)
    metrics = MetricsEvaluator(Y_test, y_pred)
    print(metrics)
    metrics.plot_predictions()

    model_tree = Pipeline([
        ('model', CustomModelTraining(model_type='tree', scaler=scaler_Y))  #Model choice: Linear regression [linear] or Decision tree [tree]
    ])
    model_tree.fit(X_train_scaled, Y_train_scaled)
    y_pred = model_tree.predict(X_test_scaled)

    metrics = MetricsEvaluator(Y_test, y_pred)
    print(metrics)
    metrics.plot_predictions()

    #Step 5: Model saving
    model_saver = ModelSaver(model_linear_regression, scaler_X, scaler_Y)
    model_saver.save()

if __name__ == "__main__":
    train_eval_model()