import pandas as pd
import mlflow
import sys
from pathlib import Path
from utils.mlflow_config import configure_mlflow

model_name = "movie-recommender"
# default version; will be overwritten with latest available
model_version = 1

def main():
    configure_mlflow()
    client = mlflow.tracking.MlflowClient()
    # try to pick the latest registered version automatically
    try:
        latest_versions = client.get_latest_versions(model_name)
        if latest_versions:
            model_version = latest_versions[0].version
            print(f"Using latest registered version {model_version}")
    except Exception as e:
        print(f"Could not query registry: {e}")

    # first try loading from registry
    model_uri = f"models:/{model_name}/{model_version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model from registry URI {model_uri}")
    except Exception as e:
        print(f"Registry load failed: {e}")
        # fall back to latest run artifact for this experiment
        print("Falling back to search for model artifact in latest run...")
        runs = mlflow.search_runs(experiment_names=["movielens-content-based"], order_by=["start_time DESC"], max_results=1)
        if runs.empty:
            raise RuntimeError("No runs found to load model from")
        run_id = runs.iloc[0].run_id
        fallback_uri = f"runs:/{run_id}/model"
        print(f"Attempting to load from {fallback_uri}")
        model = mlflow.pyfunc.load_model(fallback_uri)
        print(f"Loaded model from run {run_id}")

    # build sample input and predict
    input_df = pd.DataFrame([
        {"title": "Toy Story", "top_k": 5},
        {"title": "Jumanji", "top_k": 5},   
    ])
    
    preds = model.predict(input_df)
    
    for query, recs in zip(input_df["title"], preds):
        print(f"Recommendations for '{query}':")
        for r in recs:
            print("  -", r)
            
if __name__ == "__main__":
    main()