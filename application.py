from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from models import HousingFeatures 
from data_preprocessing import DataPreprocessing
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

MODEL_PATH = "model.pkl"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_Y.pkl"

# Load the trained model
logger.info("Loading model and scalers...")
try:
    model = joblib.load(MODEL_PATH)
    scaler_X_2 = joblib.load(SCALER_X_PATH)
    scaler_Y_2 = joblib.load(SCALER_Y_PATH)
    logger.info("Model and scalers loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or scalers: {e}")
    raise e

# API endpoint to receive input features and return predictions
@app.post("/predict/")
def predict(features: HousingFeatures):
    try:
        logger.info("Received prediction POST request")
        # Convert the input features to a DataFrame
        input_data = pd.DataFrame({'longitude':[features.longitude], 'latitude':[features.latitude], 'housing_median_age':[features.housing_median_age], \
                                'total_rooms':[features.total_rooms], 'total_bedrooms':[features.total_bedrooms], 'population':[features.population], \
                                 'households':[features.households], 'median_income':[features.median_income], 'ocean_proximity':[features.ocean_proximity]})
        
        categorical_cols = ['ocean_proximity']
        preprocessing_pipeline = Pipeline([
             ('one_hot_encoder', DataPreprocessing.CustomOneHotEncoder(columns=categorical_cols, prediction=True)),
             ('feature_eng', DataPreprocessing.FeatureEngineering(True, True, True, False)),
         ])

        data = preprocessing_pipeline.fit_transform(input_data)

        logger.info("Input data preprocessed")

        input_scaled = scaler_X_2.transform(data)

        logger.info("Input data rescaled")

        try:
            pred = model.predict(input_scaled)
            logger.info("Model prediction OK")
        except Exception as e:
            logger.error(f"Error in the model prediction: {e}")

        return {"predicted_price": [pred[0][0]]}

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")

if __name__ == "__main__":
    logger.info("Starting the FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)