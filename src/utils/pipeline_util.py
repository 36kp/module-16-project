import utils.preprocess_util as pu
import transformer_util as tu
import trainer_util as mu
import pandas as pd

def get_pipeline_steps(data: pd.DataFrame):
    return [
        {
            "name": "preprocess",
            "transformer": pu.preprocess_data(data),
        },
        {
            "name": "feature_engineering",
            "transformer": tu.transform_data(data),
        },
        {
            "name": "train",
            "transformer": mu.train_model(data),
        },
    ]
    
def run_pipeline(data: pd.DataFrame):
    steps = get_pipeline_steps(data)
    for step in steps:
        print(f"Running step: {step['name']}")
        step["transformer"]
        print(f"Step {step['name']} completed")