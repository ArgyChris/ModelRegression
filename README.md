## Model Regression on California Housing Market


**Author:** Argy Christodoulidis, Ph.D. 
 
**Date:** 10/11/2024  


## Project Overview

This project aims to develop models to predict housing values in California. The solution includes data preprocessing, model training, and a REST API to serve predictions.

---

## Project Structure

Brief description:

The purpose of this exercise was to develop models to solve the problem of predicting the house value in the state of California.

Project structure:

ModelRegression

├── **application.py**

├── **data_preprocessing_model_dev.py**

├── data_preprocessing.py

├── Dockerfile

├── **exploratory_data_analysis.ipynb**

├── model.pkl

├── models.py

├── model_training.py

├── requirements_app.txt

├── requirements_model_dev.txt

├── scaler_X.pkl

└── scaler_Y.pkl


### Contents:
1. A Jupyter Notebook for exploratory data analysis
2. Scripts for data preprocessing and model development
3. A REST API application to serve predictions

---

## Approach

### 1. Data Exploration

**File:** `exploratory_data_analysis.ipynb`  

In this step, we explored the dataset to identify any characteristics that require special handling. This included checking for missing values, data types, duplicates, and abnormal data distributions. Visualizations were also generated for better data understanding.

### 2. Data Preprocessing and Model Development

**Files:** `data_preprocessing_model_dev.py`, `data_preprocessing.py`  

The project uses custom class wrappers for scikit-learn's pipeline modules, allowing for flexible configuration and modularity.

- **Data Preprocessing:**  
  The `data_preprocessing.py` script handles various preprocessing tasks, including data downloading, imputation, one-hot encoding, feature engineering, and stratified sampling. The modular design enables reuse of these preprocessing functions in both model training and prediction serving.

- **Model Development:**  
  The `data_preprocessing_model_dev.py` script includes a dynamic model selection pipeline. Users can choose between Linear Regression and Decision Tree Regression models. The custom model class (`CustomModelTraining`) is easily extendable to include other models, such as Random Forests or SVMs.

### 3. Application

**File:** `application.py`  

A REST API application built with FastAPI is accessible locally via Swagger. The application processes client-server communication for predictions using a POST request, passing feature values as part of the payload to adhere to security standards. Our custom preprocessing pipelines are usable and here are used here to preprocess the input. The current application is simplified to accept a single input payload per request, and in the future we plan to allow the user to upload fileare the user input before scoring with the model.

### 4. Other Considerations

**File:** `Dockerfile` , `models.py` 

The application was containerized with Docker to ensure platform independence, as the development was done on a Linux environment. Detailed documentation and logging are included throughout the project to improve readability, facilitate debugging, and enhance maintainability. Also in the application we implemented the pydantic to control the API parameters input. 

---

## Installation and Running

### 1. Clone the Repository

**clone:** git clone <https://github.com/ArgyChris/ModelRegression>

cd ModelRegression

### 2i. Create Environment (python ver. 3.7.9)

**Windows:** python -m venv ModelRegression 

source ModelRegression/bin/activate  

pip install -r requirements_model_dev.txt

### 2i. Create Environment (conda in Linux)

**Linux:** conda create -n "ModelRegression" python=3.7.9

conda create -n "ModelRegression" python=3.7.9

pip install -r requirements_model_dev.txt

### 3 Run the process to retrain the model

python data_preprocessing_model_dev.py

### 4 Build and run the application from the Docker image 

**prerequisite:** Docker application is running or docker is installed in the environment

docker build -r prediction-app .

docker images (to confirm the build)

uvicorn application:app --host 0.0.0.0 --port 8000 --reload 

### 4 Access the swagger webpage  

**Windows/Linux:** http://0.0.0.0:8000/docs

---

### Future Considerations

This project lays the infrastructure for a predictive model and a serving application, but several improvements are planned for future development:

**Model Optimization and Expansion:** Currently, the model has not been optimized, and we’ve focused primarily on baseline algorithms (Linear Regression and Decision Tree). In future iterations, we aim to incorporate more advanced models, such as Random Forest, Gradient Boosting, or neural networks, and perform hyperparameter tuning to improve predictive performance.

**Enhanced API Functionality:** The current application is simplified to accept a single input payload per request, and in the future we plan to allow the user to upload files.

