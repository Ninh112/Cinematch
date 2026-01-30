mkdir -p data/raw data/processed notebooks
mkdir -p src/data src/models src/inference
touch src/data/__init__.py src/data/make_dataset.py src/models/__init__.py src/models/train_model.py  src/inference/__init__.py src/inference/predict.py notebooks/train.ipynb
touch README.md dvc.yaml requirements.txt