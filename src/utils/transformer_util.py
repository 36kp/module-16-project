import pandas as pd
def encode_and_explode_data(data: pd.DataFrame):
    # Create dummy columns for categorical columns
    dummy_columns = ['content_rating', 'color']
    for col in dummy_columns:
        print(col)
        column_dummies = pd.get_dummies(data[col], prefix=col, prefix_sep='_', dtype='int') 
        data = pd.concat([data, column_dummies], axis=1)
        data.drop(columns=[col], inplace=True)
    return explode_data(data)

def explode_data(data: pd.DataFrame):    
    data['genres'] = data['genres'].apply(lambda x: x.split('|'))
    genre_dummies = data['genres'].explode().str.get_dummies().groupby(level=0).max()
    data = pd.concat([data.drop(columns=['genres']), genre_dummies], axis=1)    
    return data

def transform_data(data: pd.DataFrame):
    return data