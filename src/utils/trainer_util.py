import pandas as pd
import utils.transformer_util as tu
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def perform_correlation_analysis(data: pd.DataFrame):
    """
    Accepts a DataFrame and returns a correlation matrix.
    """
    print("Performing correlation analysis")
    print(data.corr())
    return data.corr()

def train_model(model: Pipeline, data: pd.DataFrame, target_col: str, debug=False):
    """
    Accepts a DataFrame and returns a trained model.
    """
    print("Training model")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = create_train_test_splits(data)
    
    # Encode training dataset
    X_train = tu.encode_data(X_train)
    
    # Fit the model
    model = model.fit(X_train, y_train)
    
    # Encode test dataset
    X_test = tu.encode_data(X_test)
    
    if debug:
        perform_correlation_analysis(X_train)
        
    return X_test, y_test, model
    

def create_train_test_splits(data: pd.DataFrame):
    """
    Accepts a DataFrame and returns training and test splits.
    """
    print("Creating train and test splits")
    X = data.drop(columns=['imdb_score'])
    y = data['imdb_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test

