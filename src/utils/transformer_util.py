import pandas as pd

def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna()