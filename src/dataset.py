# src/dataset.py

import os
import pandas as pd

def load_pusht_parquet(file_path="/home/rahulpanchal7/hdd/rahul_ros2/pusht/data/chunk-000/file-000.parquet"):
    """
    Load the single PushT parquet file.

    Args:
        file_path (str): Full path to the parquet file.

    Returns:
        df (pd.DataFrame): Loaded episode data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    df = pd.read_parquet(file_path)
    return df

def main():
    file_path = "/home/rahulpanchal7/hdd/rahul_ros2/pusht/data/chunk-000/file-000.parquet"
    episode = load_pusht_parquet(file_path)

    print("Loaded episode shape:", episode.shape)
    print("\nColumns:", episode.columns)
    print("\nFirst 5 rows:\n", episode.head())

if __name__ == "__main__":
    main()
