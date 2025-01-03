import pandas as pd
import kagglehub
import os
import shutil

file_name = "movie_metadata.csv"
kaggle_data = "carolzhangdc/imdb-5000-movie-dataset"

def aquireIMDbDataFrame(path = "../resources"):
    """
    Aquires the IMDb data from a local file or from Kaggle if the file does not exist locally.

    Returns: A pandas DataFrame with the IMDb data.
    """
    resources_path = path
    file_path = os.path.join(resources_path, file_name)
    print(f"File path: {file_path}")
    print(f"Reading data from {file_path}")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print("File does not exist locally. Downloading from Kaggle.")
        downloadKaggleIMDbFile(resources_path)
        return aquireIMDbDataFrame()
    

def downloadKaggleIMDbFile(resources_path):
    """
    Copies a file from Kraggle.
    """
    
    kaggle_path = kagglehub.dataset_download(kaggle_data)
    source = os.path.join(kaggle_path, file_name)
    destination = os.path.join(resources_path, file_name)
    copy_file(source, destination)


def copy_file(source, destination):
    """
    Copies a file from source to destination.

    :param source: The path of the source file to copy.
    :param destination: The path where the file should be copied.

    Raises: section with FileNotFoundError: if source file is not found on local destination path. 
    Exception: if an unknown error occurs.
    """
    try:
        shutil.copy(source, destination)
        print(f"File copied successfully from {source} to {destination}")
    except FileNotFoundError:
        print(f"Source file not found: {source}")
    except Exception as e:
        print(f"An error occurred: {e}")