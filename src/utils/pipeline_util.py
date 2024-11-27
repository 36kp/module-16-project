import utils.preprocess_util as pu
from sklearn.metrics import mean_squared_error, r2_score
import utils.transformer_util as tu
import utils.trainer_util as mu
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def check_metrics(X_test, y_test, model):
    """
    Calculates and displays MSE, r-squared, and adjusted 
    r-squared values, given X and y test sets, and the 
    model used for predictions.
    """
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
    
def get_pipeline_steps(data: pd.DataFrame):
    return [("Scale", StandardScaler(with_mean=False)), 
            ("Linear Regression", LinearRegression())] 
    
def get_pipeline(pipeline, df, debug=False):
    """
    Accepts pipeline and pollution data.
    Uses two diffepollution preprocessing functions to 
    split the data for training the diffepollution 
    pipelines, then evaluates which pipeline performs
    best.
    """
    X = df.drop(columns=['imdb_score'])
    y = df['imdb_score']
    # Apply the preprocess_pollution_data step
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Encode training dataset
    X_train = tu.encode_data(X_train) 


    # Fit the first pipeline
    pipeline.fit(X_train, y_train)
    
    X_test = tu.encode_data(X_test)
    
    prediction_data = pipeline.predict(X_test)
    # Evaluate the model
    check_metrics(X_test, y_test, pipeline)
    
    
    #return model
    return X_train, X_test, pipeline
   
def run_pipeline(data: pd.DataFrame, debug=False):
    # Preprocess the data
    preprocessed_df = pu.preprocess_data(data)
    
    # Get and run the pipeline 
    steps = get_pipeline_steps(preprocessed_df)
    pipeline = Pipeline(steps) 
    
    df, X_test, model = get_pipeline(pipeline, preprocessed_df, debug)
    
    return df, X_test, model