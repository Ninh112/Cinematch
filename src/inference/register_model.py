from mlflow.tracking import MlflowClient
import mlflow
from pathlib import Path
import sys
from utils.mlflow_config import configure_mlflow

def get_latest_run(client):
    # 1. Get experiment id (you must know the experiment name or set it in training)
    experiment = client.get_experiment_by_name("movielens-content-based") 
    if experiment is None:
        raise RuntimeError("Experiment 'movielens' not found. Training script must create it.")
    experiment_id = experiment.experiment_id

    # 2. Fetch the latest run in this experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No runs found. Make sure train_model.py logged a run.")

    latest_run = runs[0]
    run_id = latest_run.info.run_id

    print(f"Latest run detected: {run_id}")
    
    return run_id

def main():
    configure_mlflow()
    client = MlflowClient()

    model_name = "movie-recommender"
    run_id = get_latest_run(client)

    print(f"Registering model '{model_name}' from run_id: {run_id}")
    
    source = f"runs:/{run_id}/model"
    print(f"Registering model version from: {source}")

    # 1. Ensure registered model exists
    try:
        client.get_registered_model(model_name)
        print(f"Registered model '{model_name}' already exists.")
    except Exception:
        print(f"Registered model '{model_name}' does not exist. Creating it.")
        client.create_registered_model(model_name)
        
    # 2. Create a new model version
    mv = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
    )
    print(f"Created model version: {mv.version}")
    
    # 3. set alias - optional
    
    alias = "production"
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=mv.version,
    )
    print(f"Set alias '{alias}' -> {model_name} v{mv.version}")
    
    # 4. Show all versions
    print(f"\n=== Registered model: {model_name} ===")
    info = client.get_registered_model(model_name)
    print("Model info:", info.name)
    print("Versions:")
    for mv in client.search_model_versions(f"name = '{model_name}'"):
        print(
            f"  version={mv.version}, aliases={mv.aliases}, tags={mv.tags}"
        )
        
if __name__ == "__main__":
    main()