#!/usr/bin/env python3
import pathlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow

# BASE_DIR will be like '/home/jovyan/DemoExample/'
BASE_DIR = pathlib.Path().absolute()
print(f"Working dir: {BASE_DIR}")

# Loading data
housing = fetch_california_housing(data_home=BASE_DIR)

data = pd.DataFrame(housing.data)
data.columns = housing.feature_names
data['Price'] = housing.target

data.info()

# Train/test split
target_name = "Price"
X, y = data.drop(target_name, axis=1), data[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start training
mlflow.set_tracking_uri('file:/home/jovyan/mlruns')
mlflow.set_experiment("cpu_job.ipynb")
with mlflow.start_run(nested=True) as run:

    # Train model
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)
    
    # Calculate metric & save
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mlflow.log_metric("RMSE", rmse)  # add mlflow metrics
    

# Print metric
print("=" * 50)
print(f"RMSE: {rmse}")
