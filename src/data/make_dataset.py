import os
import pandas as pd

def load_raw_movielens():
    item_paths = os.path.join("data", "raw", "ml-100k", "u.item")
    items = pd.read_csv(
        item_paths,
        sep = "|",
        header = None,
        encoding = "latin-1",
        usecols= [0, 1],
        names=["itemId", "title"]
    )
    return items

def preprocess(df):
    df = df.dropna(subset=["title"])
    return df

def save_processed(df):
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/interactions.csv"
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    df = load_raw_movielens()
    df = preprocess(df)
    save_processed(df)
    print("Saved processed data to data/processed/interactions.csv")
