from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.tracking
import pandas as pd
import sys
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from utils.mlflow_config import configure_mlflow

model_name = "movie-recommender"
model_version = 1

class RecommendRequest(BaseModel):
    title: str = "Toy Story"
    top_k: int = 5
    
class RecommendResponse(BaseModel):
    title: str
    recommendations: list[str]
    
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Model on Startup
    configure_mlflow()
    client = mlflow.tracking.MlflowClient()
    
    # try to get latest registered version
    try:
        latest_versions = client.get_latest_versions(model_name)
        if latest_versions:
            version = latest_versions[0].version
            print(f"Using latest registered version {version}")
    except Exception as e:
        print(f"Could not query registry: {e}")
        version = model_version
    
    # first try loading from registry
    model_uri = f"models:/{model_name}/{version}"
    try:
        app.state.model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model from registry URI {model_uri}")
    except Exception as e:
        print(f"Registry load failed: {e}")
        print("Falling back to latest run artifact...")
        try:
            runs = mlflow.search_runs(experiment_names=["movielens-content-based"], order_by=["start_time DESC"], max_results=1)
            if runs.empty:
                raise RuntimeError("No runs found to load model from")
            run_id = runs.iloc[0].run_id
            fallback_uri = f"runs:/{run_id}/model"
            print(f"Attempting to load from {fallback_uri}")
            app.state.model = mlflow.pyfunc.load_model(fallback_uri)
            print(f"Loaded model from run {run_id}")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            raise

    yield
    
app = FastAPI(
    title="Movie Recommender API",
    version="1.0.0",
    description="Content based movie recommender served from MLflow Model Registry.",
    lifespan=lifespan,
)

# during development, allow all origins to avoid CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],            
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    print(f"[API] /recommend called with {req}")
    model = app.state.model
    
    input_df = pd.DataFrame(
        [
            {"title": req.title, "top_k": req.top_k}
        ]
    )
    try:
        preds = model.predict(input_df)
    except Exception as e:
        print(f"[API] model prediction error: {e}")
        raise
    recommendations = preds[0] if len(preds) > 0 else []
    
    resp = RecommendResponse(
        title=req.title,
        recommendations=recommendations,
    )
    print(f"[API] returning {resp}")
    return resp