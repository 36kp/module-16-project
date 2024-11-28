import pandas as pd

def perform_correlation_analysis(data: pd.DataFrame):
    """
    Accepts a DataFrame and returns a correlation matrix.
    """
    print("Performing correlation analysis")
    print(data.corr())
    return data.corr()