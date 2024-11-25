import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    functions = [
        _genres_dummies,
        _drop, 
        _dropNaN, 
        _sort, 
        _filter, 
        _strip, 
        _fillna, 
        _strip_xa0, 
        _concat, 
        _join_explode, 
        _freq, 
        _drop_names_likes
        ]
    return _pipeline(data, functions)

def _pipeline(data: pd.DataFrame, functions: list) -> pd.DataFrame:
    for func in functions:
        data = func(data, debug = True) 
    return data


def _drop(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    drop_columns = ['movie_imdb_link','aspect_ratio', 'plot_keywords']
    processed_data = data.drop(drop_columns, axis=1)
    if debug:
        print(f"{_drop.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data

def _dropNaN(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    drop_Nan_columns = ['title_year']
    processed_data = data.dropna(subset=drop_Nan_columns)
    if debug:
        print(f"{_dropNaN.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data

def _sort(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data[sorted(data.columns)]
    if debug:
        print(f"{_sort.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data

def _filter(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data[(data['gross']>1_000) & 
                          (data['budget']>1_000) & 
                          (data['country'] == 'USA') &
                          (data['title_year']>1994)].drop(columns='country')
    if debug:
        print(f"{_filter.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _strip(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data
    for column in processed_data.columns:
        if(processed_data[column].dtype == 'object'):
            processed_data[column] = processed_data[column].fillna('unknown')
            processed_data[column] = processed_data[column].apply(lambda x: x.strip())
    if debug:
        print(f"{_strip.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _fillna(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data.fillna(-1)
    if debug:
        print(f"{_fillna.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data

def _strip_xa0(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data
    processed_data['movie_title'] = processed_data['movie_title'].apply(lambda x: x.strip('\xa0'))
    if debug:
        print(f"{_strip_xa0.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data

def _concat(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data
    processed_data['actors'] = processed_data[['actor_1_name', 'actor_2_name', 'actor_3_name']].agg(list, axis=1)
    processed_data['facebook_likes'] = processed_data[['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes']].agg(list, axis=1)
    if debug:
        print(f"{_concat.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _join_explode(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data
    processed_data = processed_data.explode(column=['actors', 'facebook_likes'])
    if debug:
        print(f"{_join_explode.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data

def _freq(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data

    frequency = processed_data['actors'].value_counts()
    processed_data['actors_Encoded'] = processed_data['actors'].map(frequency)

    frequency = processed_data['director_name'].value_counts()
    processed_data['directors_Encoded'] = processed_data['director_name'].map(frequency)
    if debug:
        print(f"{_freq.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _drop_names_likes(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    drop_columns = ['actor_1_facebook_likes', 
                    'actor_1_name', 
                    'actor_2_facebook_likes',
                    'actor_2_name', 
                    'actor_3_facebook_likes', 
                    'actor_3_name', 
                    'director_name', 
                    'actors', 
                    'movie_title']
    processed_data = data.drop(drop_columns, axis=1)
    if debug:
        print(f"{_drop_names_likes.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _genres_dummies(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    # TODO modify this with a lambda function
    genres = data
    genres['genres'] = genres['genres'].astype(str)
    genres['genres'] = genres['genres'].str.split('|')
    genre_dummies = genres['genres'].explode().str.get_dummies().groupby(level=0).max()
    processed_data = pd.concat([genres.drop(columns=['genres']), genre_dummies], axis=1)
    if debug:
        print(f"{_genres_dummies.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data
