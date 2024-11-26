import utils.preprocess_util as pu

import transformer_util as tu
import trainer_util as mu
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n = x.shape[0]
    p = y.shape[1]
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def get_pipeline_steps(data: pd.DataFrame):
    return [("Scale", StandardScaler(with_mean=False)), 
            ("Linear Regression", LinearRegression())] 
    
def get_pipeline(pipeline, df):
    """
    Accepts pipeline and pollution data.
    Uses two diffepollution preprocessing functions to 
    split the data for training the diffepollution 
    pipelines, then evaluates which pipeline performs
    best.
    """
    # Apply the preprocess_pollution_data step
    X_train, X_test, y_train, y_test = train_test_split(df, random_state=42)
    
    # Encode training dataset
    X_train = tu.encode_and_explode_data(X_train) 


    # Fit the first pipeline
    pipeline.fit(X_train, y_train)
    
    #return model
    return pipeline
   
def run_pipeline(data: pd.DataFrame):
    # Preprocess the data
    preprocessed_df = pu.preprocess_data(data)
    print(preprocessed_df.head(5))
    print(preprocessed_df.tail(5))
    
    # Get and run the pipeline 
    steps = get_pipeline_steps(preprocessed_df)
    pipeline = Pipeline(steps) 
    
    model = get_pipeline(pipeline, preprocessed_df)
    
    
    return data