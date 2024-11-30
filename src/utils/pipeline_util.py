import utils.preprocess_util as pu
from sklearn.metrics import mean_squared_error, r2_score
import utils.transformer_util as tu
import utils.trainer_util as trainer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given X and y sets 
    and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def check_metrics(X_test, y_test, model):
    """
    Calculates and displays MSE, r-squared, and adjusted 
    r-squared values, given X and y test sets, and the 
    model used for predictions.
    
    Args:
        X_test (pd.DataFrame): Test set features.
        y_test (pd.Series): Test set target values.
        model (Model): Model used for predictions.
    Returns:
        y_pred (np.array): Predicted target values.
    """
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
    print(f"Adjusted R-squared: {r2_adj(X_test, y_test, model)}")
    
    return y_pred
    
def get_pipeline_steps_PCA(data: pd.DataFrame):
    """
    Create a list of steps for the pipeline with PCA components.

    Args:
        data (pd.DataFrame): DataFrame containing the data.

    Returns:
        List: List of steps for the pipeline.
    """
    return get_pipeline_steps(data).insert(1, ("PCA", PCA(n_components=2)))
    
def get_pipeline_steps(data: pd.DataFrame, model=LinearRegression()):
    """
    Create a list of steps for the pipeline.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        model (Model, optional): Model to train in the pipeline. Defaults to LinearRegression().

    Returns:
        List: List of steps for the pipeline.
    """
    return [("Scale", StandardScaler(with_mean=False)), 
            ("Model", model)] 
    
def get_pipeline(pipeline: Pipeline, df: pd.DataFrame , debug=False):
    """
    Train and evaluate the pipeline.
    Args:
        pipeline (Pipeline): Pipeline object to train.
        df (DataFrame): DataFrame containing the data.
        debug (bool, optional): Print debug information. Defaults to False.

    Returns:
        Pipeline: Trained pipeline.
    """
    
    X_test, y_test, pipeline = trainer.train_model(pipeline, df, 'imdb_score', debug)
    
    # Evaluate the model
    y_pred = check_metrics(X_test, y_test, pipeline)
    
    #return model
    return  pipeline, y_pred

def extract_pca_components(pipeline, X, y):
    """
    Extract PCA components after pipeline transformation.
    """
    pca = pipeline.named_steps['PCA']  # Access the PCA step in the pipeline
    pca_components = pca.transform(X)  # Transform the input data using PCA
    pca_df = pd.DataFrame(
        pca_components, 
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    pca_df['imdb_score'] = y.reset_index(drop=True)
    return pca_df

def calculate_pca_correlation(pca_df):
    """
    Calculate and display the correlation matrix for PCA components.
    """
    print("Correlation Matrix of PCA Components:")
    correlation_matrix = pca_df.corr()
    print(correlation_matrix)
    return correlation_matrix

def run_pipeline_PCA(data: pd.DataFrame, debug=False):
    """
    Runs the pipeline, including preprocessing, PCA, and evaluation.
    """
    # Preprocess the data
    preprocessed_df = pu.preprocess_data(data)

    # Configure pipeline steps
    steps = get_pipeline_steps_PCA(preprocessed_df)
    pipeline = Pipeline(steps)

    # Train and evaluate the pipeline
    X_train, X_test, y_test, trained_pipeline = get_pipeline(pipeline, preprocessed_df, debug)

    # If PCA is used, extract and analyze PCA components
    pca_df = extract_pca_components(trained_pipeline, X_test, y_test)
    calculate_pca_correlation(pca_df)

    return X_train, X_test, y_test, trained_pipeline

def run_pipeline(data: pd.DataFrame, debug=False):
    """
    Runs the pipeline, including preprocessing, PCA, and evaluation.
    """
    # Preprocess the data
    preprocessed_df = pu.preprocess_data(data)

    # Configure pipeline steps
    steps = get_pipeline_steps(preprocessed_df)
    pipeline = Pipeline(steps)

    # Train and evaluate the pipeline
    trained_pipeline, y_pred = get_pipeline(pipeline, preprocessed_df, debug)

    return trained_pipeline, y_pred