## Model Regression on California Housing Market


Author: Argy Christodoulidis, Ph.D.

Date: 10/11/2024

--------------------------------------------
Brief description:

The purpose of this exercise was to develop models to solve the problem of predicting the house value in the state of California.

Structure of the project:

ModelRegression
.
├── application.py
├── data_preprocessing_model_dev.py
├── data_preprocessing.py
├── Dockerfile
├── exploratory_data_analysis.ipynb
├── Figure_1.png
├── Figure_2.png
├── model.pkl
├── models.py
├── model_training.py
├── __pycache__
│   ├── application.cpython-313.pyc
│   ├── application.cpython-37.pyc
│   ├── data_preprocessing.cpython-313.pyc
│   ├── data_preprocessing.cpython-37.pyc
│   ├── models.cpython-313.pyc
│   ├── models.cpython-37.pyc
│   ├── model_training.cpython-313.pyc
│   └── model_training.cpython-37.pyc
├── README.md
├── requirements_app.txt
├── requirements_model_dev.txt
├── scaler_X.pkl
└── scaler_Y.pkl


For this purpose the following where developed:

1) A python notebook for exploratory data analysis

2) Code to perform data preprocessing and model development

3) An application based on simple REST API to serve prediction

is
----------------------------------------------

Approach:

### Data exploration:

code in exploratory_data_analysis.ipynb

The first step was the data exploration to identify special characteristics of the dataset that require extra care and attention in order to prepare the dataset for the model development phase, such as to check for missing values, data types, duplicates, or abnormal data distributions. Also, we generated visualizations for better data interpretation.

### Data preprocessing and model development

code in data_preprocessing_model_dev.py

Both for the model development and the application we created custom classes wrappers for the scikit learn learn Pipelines modules. In that way we offer a modular solution. Each class implements different functionality or module of the data preprocessing pipeline. The different modules can be combined in modular pipeline with by controlling some input parameters in the pipelines. For the data preprocessing we have the data_preprocessing.py that implements the data downloading, and preprocessing (imputation, one-hot encoding, feature engineering, sampling). The modules are flexible and can be used both in the code for model training as well as for serving the predictions. We followed a similar approach for the model training development phase where we have provided a way for dynamic model selection that lets the user select between a Linear Regression or Decision Tree Regression model (see in model_training.py) as part of a dynamic model selection pipeline. The custom model class (CustomModelSwitcher) is flexible and can be extended to handle other models as needed, like Random Forests, SVMs, etc.


### Application

code in application.py

For the application we implemented a simple REST API application based on the FastAPI that is accessible locally via the Swagger page. We implemented the communication between the client-server for the predictions via a POST request in order to pass the feature values as part of the payload adhering to the security standards. Also, we make use of our custom pipelines to preprocess the user input before the model scoring.

### Other considerations

code in Dockerfile

All the code was developed in a Linux environment. Therefore we containerized the application with docker to be platform agnostic. We also include extensive code documentation as well as application logging to make the code and the debugging easier.


--------------------------------------------------