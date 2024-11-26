import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    functions = [
        _drop, 
        _dropNaN, 
        _sort, 
        _filter, 
        _strip, 
        _fillna, 
        _concat, 
        _join_explode, 
        _freq
        ]
    return _pipeline(data, functions)

def _pipeline(data: pd.DataFrame, functions: list) -> pd.DataFrame:
    for func in functions:
        data = func(data) 
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

def _concat(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data
    processed_data['actors'] = processed_data[['actor_1_name', 'actor_2_name', 'actor_3_name']].agg(list, axis=1)
    processed_data['actor_facebook_likes'] = processed_data[['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes']].agg(list, axis=1)
    if debug:
        print(f"{_concat.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _join_explode(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data
    processed_data = processed_data.explode(column=['actors', 'actor_facebook_likes'])
    processed_data.reset_index(drop=True, inplace=True) 
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



# TODO MOVE THIS TO A SEPARATE FILE

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



def bucket_contentRatings(data: pd.DataFrame) -> pd.DataFrame:
    content_rating_df = data[['content_rating']].copy()
    content_rating_df['content_rating'] = content_rating_df['content_rating'].fillna("other")
    total_count = content_rating_df['content_rating'].value_counts().sum()
    content_rating_df['percentage'] = content_rating_df['content_rating'].map(content_rating_df['content_rating'].value_counts()) / total_count * 100
    content_rating_df["rating_bin"] = content_rating_df["content_rating"].where(content_rating_df["percentage"] >= 10, "other")
    content_rating_df.drop(columns=['content_rating','percentage'], inplace=True)
    return pd.concat([data, content_rating_df], axis=1)


def process_genres(data: pd.DataFrame) -> pd.DataFrame:
    data['genres'] = data['genres'].fillna("other_genre")
    data['genres'] = data['genres'].str.split('|')
    all_genres = [genre for sublist in data['genres'] for genre in sublist]
    genre_counts = pd.Series(all_genres).value_counts()
    threshold = len(data) * 0.1
    frequent_genres = genre_counts[genre_counts > threshold].index
    for genre in frequent_genres:
        data[genre] = data['genres'].apply(lambda x: genre in x).astype(int)

    data['other_genre'] = data['genres'].apply(lambda x: any(genre not in frequent_genres for genre in x)).astype(int)
    data = data.drop(columns=['genres'])
    return data



def director_frequence(data: pd.DataFrame) -> pd.DataFrame:
    data['director_name'] = data['director_name'].fillna('unknown_director')
    director_frequencies = data['director_name'].value_counts()
    data['director_frequency'] = data['director_name'].map(director_frequencies)
    data = data.drop(columns=['director_name'])
    return data