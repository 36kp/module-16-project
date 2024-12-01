import utils.preprocess_util as pu
from sklearn.metrics import mean_squared_error, r2_score
import utils.transformer_util as tu
import utils.trainer_util as trainer
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given X and y sets 
    and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def check_metrics(X_test, y_test, model, debug=False):
    """
    Calculates and displays MSE, r-squared, and adjusted 
    r-squared values, given X and y test sets, and the 
    model used for predictions.
    
    Args:
        X_test (pd.DataFrame): Test set features.
        y_test (pd.Series): Test set target values.
        model (Model): Model used for predictions.
        debug (bool, optional): Print debug information. Defaults
    Returns:
        y_pred (np.array): Predicted target values.
    """
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)
    
    # Create a dictionary to store the metrics
    metrics = {
        "Mean Squared Error": mean_squared_error(y_test, y_pred),
        "R-squared": r2_score(y_test, y_pred),
        "Adjusted R-squared": r2_adj(X_test, y_test, model)
    }

    # Print out the MSE, r-squared, and adjusted r-squared values
    if debug:
        print("Mean Squared Error:", metrics["Mean Squared Error"])
        print("R-squared:", metrics["R-squared"]) 
        print("Adjusted R-squared:", metrics["Adjusted R-squared"])
    
    return y_pred, metrics
    
def get_pipeline_steps(data: pd.DataFrame, use_PCA = False, model=LinearRegression()):
    """
    Create a list of steps for the pipeline.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        model (Model, optional): Model to train in the pipeline. Defaults to LinearRegression().

    Returns:
        List: List of steps for the pipeline.
    """
    cols_to_scale = ['actor_total_facebook_likes',
       'budget', 'cast_total_facebook_likes', 'director_facebook_likes',
       'director_frequency', 'facenumber_in_poster', 'gross',
       'movie_facebook_likes', 'num_critic_for_reviews',
       'num_user_for_reviews', 'num_voted_users', 'total_actor_frequency']
    cols_to_exclude = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy',
       'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'duration', 'other_genre', 'rating_bin',
       'title_year']
    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), cols_to_scale),
            ('exclude', 'passthrough', cols_to_exclude)
        ]
    )
    
    # Create a list of tuples containing the steps
    steps = []
    
    steps.append(("Scale", preprocessor)) # Scale the data
    if use_PCA:
        steps.append(("PCA", PCA(n_components=0.95))) # Use PCA
    steps.append(("Model", model)) # Model to train
    
    return steps
    
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
    y_pred, metrics = check_metrics(X_test, y_test, pipeline)
    
    #return model
    return  pipeline, y_pred, metrics

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

def run_pipeline(data: pd.DataFrame, use_PCA=False, debug=False):
    """
    Runs the pipeline, including preprocessing and evaluation.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        use_PCA (bool, optional): Use PCA components. Defaults to False.
        debug (bool, optional): Print debug information. Defaults to False.
    Returns:
        Pipeline: Best of the trained pipelines.
        y_pred (np.array): Predicted target values.
    """
    # Create a dictionary of models to train and compare
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(),
        "Ridge Regression": Ridge(),
        "Support Vector Machine": SVR(),
        "Random Forest": RandomForestRegressor(n_estimators=128),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=128)
        
    }
    # Preprocess the data
    preprocessed_df = pu.preprocess_data(data)

    # A variable to store the best r2_adj value and the best model
    best_r2_adj = 0
    best_model = None
    best_pipeline = None
    best_y_pred = None
    
    # For each model in the dictionary train and evaluate the pipeline
    for model_name, model in models.items():
        # Configure pipeline steps
        steps = get_pipeline_steps(data = preprocessed_df, use_PCA=use_PCA, model=model)
        pipeline = Pipeline(steps)

        # Train and evaluate the pipeline
        trained_pipeline, y_pred, metrics = get_pipeline(pipeline, preprocessed_df, debug)
        print(f"Metrics for {model_name}: {metrics}")
        if metrics["Adjusted R-squared"] > best_r2_adj:
            best_r2_adj = metrics["Adjusted R-squared"]
            best_model = model_name
            best_pipeline = trained_pipeline
            best_y_pred = y_pred
    
    if best_r2_adj == 0:
        print("No model performed well.")
        return None
    
    print(f"The best model is {best_model} with an adjusted R-squared of {best_r2_adj}")    
    
    return best_pipeline, best_y_pred, preprocessed_df