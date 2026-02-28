import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from pathlib import Path
import sys
from utils.mlflow_config import configure_mlflow

def load_data():
    # path relative to project root
    ratings_path = os.path.join("data", "raw", "ml-100k", "u.data")
    df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1"
    )
    return df

def build_baseline(df):
    #predict global mean rating
    global_mean = df["rating"].mean()
    return global_mean

def evaluate_baseline(df, global_mean):
    # Split to demonstrate evaluation
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    y_true = test["rating"].values
    y_pred = [global_mean] * len(y_true)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

if __name__ == "__main__":
    configure_mlflow()
    mlflow.set_experiment("MovieRecommender_Baseline")
    
    with mlflow.start_run(run_name="v1"):
        df = load_data()
        global_mean = build_baseline(df)
        rmse = evaluate_baseline(df, global_mean)
        
        print(f"Global Mean Rating: {global_mean}")
        print(f"Baseline RMSE: {rmse}")
        
        mlflow.log_param("model_type", "global_mean")
        mlflow.log_metric("rmse", rmse)

        mlflow.log_artifact("requirements.txt")
        
        print("Yeee!!!.. Experiment run completed and logged to MLflow.")