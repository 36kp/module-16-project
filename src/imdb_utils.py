import pandas as pd
import kagglehub
import os
import shutil

resources_path = "../resources"
file_name = "movie_metadata.csv"
def aquireIMDbDataFrame():
    """
    Aquires the IMDb data from a local file or from Kaggle if the file does not exist locally.
    """

    file_path = os.path.join(resources_path, file_name)
    print(f"Reading data from {file_path}")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print("File does not exist locally. Downloading from Kaggle.")
        downloadKaggleIMDbFile()
        return aquireIMDbDataFrame()
    

def downloadKaggleIMDbFile():
    """
    Copies a file from Kraggle.
    """
    
    kaggle_path = kagglehub.dataset_download("carolzhangdc/imdb-5000-movie-dataset")
    source = os.path.join(kaggle_path, file_name)
    destination = os.path.join(resources_path, file_name)
    copy_file(source, destination)


def copy_file(source, destination):
    """
    Copies a file from source to destination.

    :param source: The path of the source file to copy.
    :param destination: The path where the file should be copied.
    """
    try:
        shutil.copy(source, destination)
        print(f"File copied successfully from {source} to {destination}")
    except FileNotFoundError:
        print(f"Source file not found: {source}")
    except Exception as e:
        print(f"An error occurred: {e}")