import pandas as pd



def encode_data(data: pd.DataFrame):
    # LabelEncode rating_bin column
    data['rating_bin'] = data['rating_bin'].astype('category')
    data['rating_bin'] = data['rating_bin'].cat.codes
    return data
